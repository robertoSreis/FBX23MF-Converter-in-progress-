def build_with_lib3mf(self, vertices, faces, colors, palette, out_3mf):
        """Constrói 3MF com lib3mf - cores aplicadas durante criação"""
        import lib3mf as Lib3MF
        import numpy as np
        
        self.log("  Criando modelo 3MF...")
        wrapper = Lib3MF.Wrapper()
        model = wrapper.CreateModel()
        
        # Criar materiais da paleta
        self.log("  Criando materiais...")
        base_materials = model.AddBaseMaterialGroup()
        
        palette_int = (palette * 255).astype(np.uint8)
        for i, color in enumerate(palette_int):
            base_materials.AddMaterial(
                f"Color{i}",
                Lib3MF.sColor(int(color[0]), int(color[1]), int(color[2]), 255)
            )
        
        # Criar mesh object
        mesh_obj = model.AddMeshObject()
        mesh_obj.SetName("colored_model")
        
        # Adicionar vértices
        self.log("  Adicionando vértices...")
        for v in vertices:
            mesh_obj.AddVertex(Lib3MF.sPosition(float(v[0]), float(v[1]), float(v[2])))
        
        # Adicionar triângulos COM CORES simultaneamente
        self.log("  Adicionando triângulos com cores...")
        
        # Verificar se colors são índices (novo método) ou RGB (antigo)
        if colors.shape == (len(faces), 3) and colors.dtype in [np.int32, np.int64]:
            # NOVO: índices diretos da paleta
            self.log("  Modo rápido: índices de paleta")
            batch_log = 50000
            for i, (face, color_indices) in enumerate(zip(faces, colors)):
                if i % batch_log == 0 and i > 0:
                    self.log(f"    {i}/{len(faces)} ({(i/len(faces)*100):.1f}%)")
                
                tri = Lib3MF.sTriangle(int(face[0]), int(face[1]), int(face[2]))
                mesh_obj.AddTriangle(tri)
                mesh_obj.SetTriangleProperties(i, base_materials, 
                    int(color_indices[0]), int(color_indices[1]), int(color_indices[2]))
        else:
            # ANTIGO: cores RGB que precisam ser mapeadas
            self.log("  Modo compatibilidade: mapeando RGB")
            colors_int = (colors * 255).astype(np.uint8)
            color_to_idx = {}
            for i, p_color in enumerate(palette):
                key = tuple((p_color * 255).astype(np.uint8))
                color_to_idx[key] = i
            
            batch_log = 25000
            for i, (face, face_colors) in enumerate(zip(faces, colors_int)):
                if i % batch_log == 0 and i > 0:
                    self.log(f"    {i}/{len(faces)} ({(i/len(faces)*100):.1f}%)")
                
                tri = Lib3MF.sTriangle(int(face[0]), int(face[1]), int(face[2]))
                mesh_obj.AddTriangle(tri)
                
                mat_ids = [color_to_idx.get(tuple(vc), 0) for vc in face_colors]
                mesh_obj.SetTriangleProperties(i, base_materials, mat_ids[0], mat_ids[1], mat_ids[2])
        
        self.log("  Finalizando 3MF...")
        model.AddBuildItem(mesh_obj, wrapper.GetIdentityTransform())
        writer = model.QueryWriter("3mf")
        writer.WriteToFile(out_3mf)
        self.log("  ✓ 3MF criado com sucesso")

import sys
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
import time
import json

from PyQt6 import QtWidgets, QtGui, QtCore

"""
FBX → 3MF Colorido - Conversor para Impressão 3D Multicolorida

DEPENDÊNCIAS:
- PyQt6: pip install PyQt6
- trimesh: pip install trimesh
- numpy: pip install numpy
- scikit-learn: pip install scikit-learn
- Lib3MF (RECOMENDADO para cores funcionarem no slicer): pip install Lib3MF

IMPORTANTE:
- Lib3MF é ESSENCIAL para que cores apareçam corretamente em Orca/Bambu/Anycubic Slicer
- Sem Lib3MF, o arquivo 3MF será gerado mas as cores podem não ser reconhecidas
"""

# ========== Helper: run subprocess and stream output ==========
def run_process_stream(cmd, on_line=None, timeout=None, creationflags=0):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                        text=True, encoding='utf-8', errors='replace', 
                        creationflags=creationflags)
    stdout_lines = []
    stderr_acc = ""
    start = time.time()
    try:
        while True:
            line = proc.stdout.readline()
            if line:
                line = line.rstrip("\n")
                stdout_lines.append(line)
                if on_line:
                    try:
                        on_line(line)
                    except:
                        pass
            else:
                if proc.poll() is not None:
                    break
                if timeout and (time.time() - start) > timeout:
                    proc.kill()
                    return proc.returncode, "\n".join(stdout_lines), "[timeout]"
                time.sleep(0.01)
        try:
            stderr_acc = proc.stderr.read() or ""
        except:
            stderr_acc = ""
        return proc.returncode, "\n".join(stdout_lines), stderr_acc
    except Exception as e:
        try:
            proc.kill()
        except:
            pass
        return -1, "\n".join(stdout_lines), str(e)

# ========== Blender script - Extração de Cores ==========
BLENDER_COLOR_EXTRACT_SCRIPT = r"""
import bpy, sys, json
from mathutils import Color

def find_base_color_image(material):
    if not material or not material.use_nodes:
        return None
    nodes = material.node_tree.nodes
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            input = node.inputs.get('Base Color')
            if input and input.is_linked:
                for link in input.links:
                    from_node = link.from_node
                    if from_node.type == 'TEX_IMAGE':
                        return getattr(from_node, "image", None)
    for node in nodes:
        if node.type == 'TEX_IMAGE':
            return getattr(node, "image", None)
    return None

def find_principled_base_color(material):
    if not material or not material.use_nodes:
        return None
    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            input = node.inputs.get('Base Color')
            if input and not input.is_linked:
                val = input.default_value[:3]
                return [float(v) for v in val]
    return None

def extract_image_colors(img, max_samples=20000):
    '''Extrai cores de uma imagem de forma mais robusta - AMOSTRAGEM MASSIVA'''
    if not img or not img.pixels:
        return []
    
    width = img.size[0]
    height = img.size[1]
    total_pixels = width * height
    
    print(f"  Imagem: {{img.name}} ({{width}}x{{height}}, {{total_pixels}} pixels)", flush=True)
    
    # Converter pixels para lista se necessário
    pixels = list(img.pixels)
    
    colors = []
    
    # SE a textura é pequena, processar TODOS os pixels
    if total_pixels <= max_samples:
        print(f"  Processando TODOS os {{total_pixels}} pixels", flush=True)
        step = 1
    else:
        # Calcular step para max_samples
        step = max(1, int((total_pixels / max_samples) ** 0.5))
        print(f"  Amostragem: step={{step}}", flush=True)
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            idx = (y * width + x) * 4
            
            if idx + 3 < len(pixels):
                r = pixels[idx]
                g = pixels[idx + 1]
                b = pixels[idx + 2]
                a = pixels[idx + 3]
                
                # Ignorar apenas pixels completamente transparentes
                if a > 0.01:
                    colors.append([r, g, b])
    
    print(f"  Extraídas {{len(colors)}} amostras", flush=True)
    return colors

try:
    print("=== INICIANDO EXTRAÇÃO DE CORES ===", flush=True)
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath="{fbx_path}")
    
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objects:
        print("ERROR: no mesh", flush=True)
        sys.exit(1)
    
    print(f"Encontrados {{len(mesh_objects)}} meshes", flush=True)
    
    # Coletar todas as cores
    all_colors = []
    
    for obj_idx, obj in enumerate(mesh_objects):
        print(f"Processando objeto {{obj_idx+1}}/{{len(mesh_objects)}}: {{obj.name}}", flush=True)
        
        if not obj.material_slots:
            print("  Sem materiais", flush=True)
            continue
        
        for slot_idx, slot in enumerate(obj.material_slots):
            mat = slot.material
            if not mat:
                print(f"  Material {{slot_idx}}: None", flush=True)
                continue
            
            print(f"  Material {{slot_idx}}: {{mat.name}}", flush=True)
            
            # 1. Tentar pegar imagem (PRIORIDADE)
            img = find_base_color_image(mat)
            if img:
                print(f"    Textura encontrada: {{img.name}}", flush=True)
                img_colors = extract_image_colors(img, max_samples=20000)
                all_colors.extend(img_colors)
            else:
                print("    Sem textura", flush=True)
                
                # 2. Fallback: cor base do material
                base_col = find_principled_base_color(mat)
                if base_col:
                    print(f"    Cor base: {{base_col}}", flush=True)
                    all_colors.append(base_col)
                else:
                    print("    Sem cor base", flush=True)
    
    print(f"\nTotal de amostras coletadas: {{len(all_colors)}}", flush=True)
    
    # Retornar cores como JSON
    result = {{
        "colors": all_colors,
        "count": len(all_colors)
    }}
    
    print("COLOR_DATA:" + json.dumps(result), flush=True)
    print("=== EXTRAÇÃO CONCLUÍDA ===", flush=True)
    sys.exit(0)
    
except Exception as e:
    import traceback
    print("=== ERRO NA EXTRAÇÃO ===", flush=True)
    traceback.print_exc()
    sys.exit(1)
"""

# ========== Blender script - Conversão DIRETA E RÁPIDA ==========
BLENDER_CONVERT_PALETTE_SCRIPT = r'''
import bpy, sys, json
import numpy as np

def log(msg):
    print(msg, flush=True)

def get_material_color(material):
    """Pega cor do material de forma simples"""
    if not material or not material.use_nodes:
        return np.array([0.8, 0.8, 0.8], dtype=np.float32)
    
    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            base_color = node.inputs.get('Base Color')
            if base_color and not base_color.is_linked:
                return np.array(base_color.default_value[:3], dtype=np.float32)
    
    return np.array([0.8, 0.8, 0.8], dtype=np.float32)

def closest_color_idx(color, palette):
    """Acha índice da cor mais próxima na paleta"""
    distances = np.sum((palette - color) ** 2, axis=1)
    return np.argmin(distances)

try:
    log("=== CONVERSÃO RÁPIDA FBX → 3MF ===")
    
    palette = {palette_json}
    palette_np = np.array(palette, dtype=np.float32)
    log(f"Paleta: {{len(palette)}} cores")
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    log("Importando FBX...")
    bpy.ops.import_scene.fbx(filepath="{fbx_path}")
    
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objects:
        log("ERROR: no mesh")
        sys.exit(2)
    
    # Juntar meshes
    bpy.ops.object.select_all(action='DESELECT')
    for o in mesh_objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    
    if len(mesh_objects) > 1:
        log("Juntando meshes...")
        bpy.ops.object.join()
    
    obj = bpy.context.active_object
    mesh = obj.data
    n_verts = len(mesh.vertices)
    n_polys = len(mesh.polygons)
    log(f"Mesh: {{n_verts}} vértices, {{n_polys}} polígonos")
    
    # ESTRATÉGIA RÁPIDA: Mapear material → cor da paleta
    log("Mapeando materiais para paleta...")
    mat_to_palette_idx = {{}}
    
    for i, slot in enumerate(obj.material_slots):
        mat = slot.material
        if mat:
            mat_color = get_material_color(mat)
            palette_idx = closest_color_idx(mat_color, palette_np)
            mat_to_palette_idx[i] = palette_idx
            log(f"  Material {{i}} ({{mat.name}}): Cor {{palette_idx}} - {{'#' + ''.join(f'{{int(c*255):02X}}' for c in palette_np[palette_idx])}}")
        else:
            mat_to_palette_idx[i] = 0
    
    # Extrair geometria RAPIDAMENTE
    log("Extraindo geometria...")
    
    vertices = np.zeros((n_verts, 3), dtype=np.float32)
    mesh.vertices.foreach_get('co', vertices.ravel())
    
    # Extrair faces e suas cores
    faces_list = []
    face_colors_list = []  # (n_faces, 3) - índice de paleta para cada vértice
    
    log("Processando faces...")
    for poly in mesh.polygons:
        n_verts_poly = len(poly.vertices)
        mat_idx = poly.material_index
        palette_idx = mat_to_palette_idx.get(mat_idx, 0)
        
        if n_verts_poly == 3:
            # Triângulo
            faces_list.append([poly.vertices[0], poly.vertices[1], poly.vertices[2]])
            face_colors_list.append([palette_idx, palette_idx, palette_idx])
        elif n_verts_poly == 4:
            # Quad → 2 triângulos
            v = poly.vertices
            faces_list.append([v[0], v[1], v[2]])
            faces_list.append([v[0], v[2], v[3]])
            face_colors_list.append([palette_idx, palette_idx, palette_idx])
            face_colors_list.append([palette_idx, palette_idx, palette_idx])
    
    faces = np.array(faces_list, dtype=np.int32)
    face_palette_indices = np.array(face_colors_list, dtype=np.int32)
    
    log(f"Total faces trianguladas: {{len(faces)}}")
    
    # Salvar dados
    log("Salvando dados...")
    np.savez_compressed(
        "{npz_path}",
        vertices=vertices,
        faces=faces,
        face_palette_indices=face_palette_indices,
        palette=palette_np
    )
    
    log("✓ Dados prontos para construção do 3MF")
    sys.exit(0)
    
except Exception as e:
    import traceback
    traceback.print_exc()
    log(f"ERROR: {{e}}")
    sys.exit(3)
'''

# ========== Preview Script ==========
BLENDER_PREVIEW_SCRIPT = r'''
import bpy, sys, math

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath="{fbx_path}")
    
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        sys.exit(1)
    
    all_coords = []
    for obj in meshes:
        for v in obj.data.vertices:
            all_coords.append(obj.matrix_world @ v.co)
    
    if all_coords:
        min_x = min(co.x for co in all_coords)
        max_x = max(co.x for co in all_coords)
        min_y = min(co.y for co in all_coords)
        max_y = max(co.y for co in all_coords)
        min_z = min(co.z for co in all_coords)
        max_z = max(co.z for co in all_coords)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        max_dim = max(max_x - min_x, max_y - min_y, max_z - min_z)
    else:
        center_x = center_y = center_z = 0
        max_dim = 2
    
    light_data = bpy.data.lights.new("Sun", type='SUN')
    light_obj = bpy.data.objects.new("Sun", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (center_x + max_dim, center_y - max_dim, center_z + max_dim)
    light_data.energy = 3.0
    
    cam = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    distance = max_dim * 2.0
    cam_obj.location = (center_x + distance * 0.7, center_y - distance * 0.7, center_z + distance * 0.5)
    
    import mathutils
    direction = mathutils.Vector((center_x, center_y, center_z)) - mathutils.Vector(cam_obj.location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = {res_x}
    scene.render.resolution_y = {res_y}
    scene.render.filepath = "{out_path}"
    scene.render.film_transparent = False
    
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.9, 0.9, 0.9, 1.0)
    
    scene.eevee.taa_render_samples = 16
    
    bpy.ops.render.render(write_still=True)
    sys.exit(0)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

# ========== Color Palette Widget ==========
class ColorButton(QtWidgets.QPushButton):
    color_changed = QtCore.pyqtSignal(tuple)
    
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(40, 40)
        self.update_style()
        self.clicked.connect(self.choose_color)
    
    def update_style(self):
        r, g, b = [int(c * 255) for c in self.color[:3]]
        # Estilo isolado - não contamina o resto da UI
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r},{g},{b}); 
                border: 2px solid #555;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid #000;
            }}
        """)
    
    def choose_color(self):
        r, g, b = [int(c * 255) for c in self.color[:3]]
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(r, g, b), self)
        if color.isValid():
            self.color = (color.redF(), color.greenF(), color.blueF())
            self.update_style()
            self.color_changed.emit(self.color)

class PaletteWidget(QtWidgets.QWidget):
    palette_changed = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.colors = []
        self.color_widgets = []
        
        title = QtWidgets.QLabel("Paleta de Cores Detectada")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        self.layout.addWidget(title)
        
        # Scroll area para muitas cores
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(400)
        
        self.grid_container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_container)
        self.grid.setSpacing(4)
        scroll.setWidget(self.grid_container)
        self.layout.addWidget(scroll)
        
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("+ Adicionar Cor")
        self.btn_remove = QtWidgets.QPushButton("✕ Remover Selecionadas")
        self.btn_reset = QtWidgets.QPushButton("↻ Resetar")
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_reset)
        self.layout.addLayout(btn_layout)
        
        self.btn_add.clicked.connect(self.add_color)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_reset.clicked.connect(self.reset_palette)
        
        self.original_colors = []
    
    def set_palette(self, colors):
        """Define a paleta de cores"""
        self.colors = [tuple(c[:3]) for c in colors]
        self.original_colors = self.colors.copy()
        self.rebuild_ui()
    
    def rebuild_ui(self):
        # Limpar grid
        for i in reversed(range(self.grid.count())): 
            self.grid.itemAt(i).widget().setParent(None)
        
        self.color_widgets = []
        
        for idx, color in enumerate(self.colors):
            row = idx // 6  # 6 colunas
            col = idx % 6
            
            container = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout(container)
            container_layout.setContentsMargins(2, 2, 2, 2)
            container_layout.setSpacing(2)
            
            # Checkbox no topo
            checkbox = QtWidgets.QCheckBox()
            checkbox.setStyleSheet("margin: 2px;")
            container_layout.addWidget(checkbox, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            
            # Botão de cor menor
            btn = ColorButton(color)
            btn.setFixedSize(40, 40)  # Reduzido de 60x60
            btn.color_changed.connect(lambda c, i=idx: self.update_color(i, c))
            container_layout.addWidget(btn)
            
            # Label hex menor
            hex_label = QtWidgets.QLabel(self.rgb_to_hex(color))
            hex_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            hex_label.setStyleSheet("font-size: 8px;")
            container_layout.addWidget(hex_label)
            
            self.grid.addWidget(container, row, col)
            self.color_widgets.append((btn, hex_label, checkbox))
    
    def rgb_to_hex(self, color):
        r, g, b = [int(c * 255) for c in color[:3]]
        return f"#{r:02X}{g:02X}{b:02X}"
    
    def update_color(self, idx, color):
        if 0 <= idx < len(self.colors):
            self.colors[idx] = color
            self.color_widgets[idx][1].setText(self.rgb_to_hex(color))
            self.palette_changed.emit()
    
    def remove_selected(self):
        # Remover cores marcadas (de trás para frente)
        for idx in reversed(range(len(self.color_widgets))):
            if self.color_widgets[idx][2].isChecked():
                self.colors.pop(idx)
        
        if len(self.colors) < 1:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Precisa ter pelo menos 1 cor!")
            self.colors = self.original_colors.copy()
        
        self.rebuild_ui()
        self.palette_changed.emit()
    
    def add_color(self):
        if len(self.colors) >= 24:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Máximo de 24 cores!")
            return
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(128, 128, 128), self)
        if color.isValid():
            self.colors.append((color.redF(), color.greenF(), color.blueF()))
            self.rebuild_ui()
            self.palette_changed.emit()
    
    def reset_palette(self):
        self.colors = self.original_colors.copy()
        self.rebuild_ui()
        self.palette_changed.emit()
    
    def get_palette(self):
        return self.colors

# ========== Worker Threads ==========
class ColorExtractThread(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    colors_ready = QtCore.pyqtSignal(list)
    
    def __init__(self, blender_path, fbx_path, max_colors=8, parent=None):
        super().__init__(parent)
        self.blender_path = blender_path
        self.fbx_path = fbx_path
        self.max_colors = max_colors
    
    def run(self):
        tmpdir = None
        try:
            self.log("Extraindo cores do modelo...")
            tmpdir = tempfile.mkdtemp(prefix="fbx_colors_")
            
            script = BLENDER_COLOR_EXTRACT_SCRIPT.format(fbx_path=self.fbx_path.replace("\\","/"))
            script_file = os.path.join(tmpdir, "extract_colors.py")
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script)
            
            cmd = [self.blender_path, "--background", "--python", script_file]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            
            rc, out, err = run_process_stream(cmd, timeout=120, creationflags=creationflags)
            
            if rc != 0:
                raise RuntimeError("Falha ao extrair cores")
            
            # Procurar JSON no output
            color_data = None
            for line in out.split('\n'):
                if line.startswith("COLOR_DATA:"):
                    import json
                    color_data = json.loads(line[11:])
                    break
            
            if not color_data or not color_data['colors']:
                self.log("Nenhuma cor encontrada, usando cores padrão")
                colors = [(0.8, 0.5, 0.3), (0.3, 0.5, 0.8)]
            else:
                # Usar K-means para agrupar cores
                self.log(f"Agrupando {color_data['count']} amostras em {self.max_colors} cores...")
                colors = self.cluster_colors(color_data['colors'], self.max_colors)
            
            self.log(f"{len(colors)} cores detectadas")
            self.colors_ready.emit(colors)
            
        except Exception as e:
            self.log(f"Erro: {e}")
            # Fallback: cores padrão
            self.colors_ready.emit([(0.8, 0.5, 0.3), (0.3, 0.5, 0.8)])
        finally:
            if tmpdir:
                try:
                    shutil.rmtree(tmpdir)
                except:
                    pass
    
    def cluster_colors(self, colors, n_colors):
        """Agrupa cores priorizando DIVERSIDADE (não frequência)"""
        try:
            import numpy as np
            
            arr = np.array(colors)
            if len(arr) == 0:
                return [(0.8, 0.5, 0.3)]
            
            self.log(f"  Analisando {len(arr)} amostras de cor...")
            
            # 1. QUANTIZAR para reduzir ruído (bins menores = mais sensível)
            # Usando bins de 0.01 (100 níveis) ao invés de 0.02
            quantized = np.round(arr / 0.01) * 0.01
            quantized = np.clip(quantized, 0, 1)
            
            # 2. Pegar cores únicas (sem considerar frequência)
            unique_colors = np.unique(quantized, axis=0)
            self.log(f"  {len(unique_colors)} cores únicas detectadas")
            
            if len(unique_colors) <= n_colors:
                result = [tuple(c) for c in unique_colors]
                self.log(f"  ✓ Todas as {len(result)} cores preservadas")
                return result
            
            # 3. ALGORITMO DE MÁXIMA DIVERSIDADE
            # Selecionar cores que são MAIS DIFERENTES entre si
            
            selected = []
            remaining = list(unique_colors)
            
            # Começar com a cor mais escura
            brightness = np.mean(remaining, axis=1)
            darkest_idx = np.argmin(brightness)
            selected.append(remaining[darkest_idx])
            remaining = np.delete(remaining, darkest_idx, axis=0)
            
            # Para cada slot restante, escolher a cor MAIS DISTANTE das já selecionadas
            while len(selected) < n_colors and len(remaining) > 0:
                selected_arr = np.array(selected)
                
                # Calcular distância mínima de cada cor restante para as já selecionadas
                min_distances = []
                for candidate in remaining:
                    # Distância euclidiana no espaço RGB
                    distances = np.sqrt(np.sum((selected_arr - candidate)**2, axis=1))
                    min_dist = np.min(distances)
                    min_distances.append(min_dist)
                
                # Escolher a cor com MAIOR distância mínima (mais diferente)
                best_idx = np.argmax(min_distances)
                selected.append(remaining[best_idx])
                remaining = np.delete(remaining, best_idx, axis=0)
            
            result = [tuple(c) for c in selected]
            
            self.log(f"  ✓ {len(result)} cores com máxima diversidade selecionadas")
            
            # Mostrar as cores no log
            for i, color in enumerate(result):
                hex_color = "#{:02X}{:02X}{:02X}".format(
                    int(color[0]*255), int(color[1]*255), int(color[2]*255)
                )
                rgb_str = f"({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
                self.log(f"    Cor {i+1}: {hex_color} {rgb_str}")
            
            return result
            
        except Exception as e:
            import traceback
            self.log(f"Erro no clustering: {e}\n{traceback.format_exc()}")
            return [(0.8, 0.5, 0.3), (0.3, 0.5, 0.8)]
    
    def log(self, msg):
        self.log_signal.emit(f"[{time.strftime('%H:%M:%S')}] {msg}")

class ConvertThread(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool, str)
    
    def __init__(self, blender_path, fbx_path, out_3mf, palette, parent=None):
        super().__init__(parent)
        self.blender_path = blender_path
        self.fbx_path = fbx_path
        self.out_3mf = out_3mf
        self.palette = palette
    
    def run(self):
        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp(prefix="fbx_convert_")
            npz_path = os.path.join(tmpdir, "mesh_data.npz")
            
            # Converter paleta para JSON
            palette_json = json.dumps(self.palette)
            
            script = BLENDER_CONVERT_PALETTE_SCRIPT.format(
                fbx_path=self.fbx_path.replace("\\","/"),
                npz_path=npz_path.replace("\\","/"),
                out_3mf=self.out_3mf.replace("\\","/"),
                palette_json=palette_json
            )
            
            script_file = os.path.join(tmpdir, "convert.py")
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script)
            
            self.log("Extraindo geometria e cores...")
            cmd = [self.blender_path, "--background", "--python", script_file]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            
            def on_line(line):
                self.log(f"  {line}")
            
            rc, out, err = run_process_stream(cmd, on_line=on_line, timeout=600, creationflags=creationflags)
            
            if rc != 0 or not os.path.exists(npz_path):
                raise RuntimeError(f"Extração falhou (code {rc})")
            
            # Construir 3MF diretamente
            self.log("Construindo 3MF com cores...")
            self.build_3mf_direct(npz_path, self.out_3mf)
            
            if os.path.exists(self.out_3mf):
                size_kb = os.path.getsize(self.out_3mf) / 1024.0
                self.log(f"✓ Pronto: {size_kb:.1f} KB")
                self.finished_signal.emit(True, f"Conversão concluída! {size_kb:.1f} KB")
            else:
                raise RuntimeError("Falha ao gerar 3MF")
            
        except Exception as e:
            import traceback
            self.log(f"ERRO: {e}\n{traceback.format_exc()}")
            self.finished_signal.emit(False, str(e))
        finally:
            if tmpdir:
                try:
                    shutil.rmtree(tmpdir)
                except:
                    pass
    
    def build_3mf_direct(self, npz_path, out_3mf):
        """Constrói 3MF diretamente dos dados - OTIMIZADO"""
        import numpy as np
        
        try:
            import Lib3MF
            use_lib3mf = True
            self.log("  Usando lib3mf")
        except ImportError:
            use_lib3mf = False
            self.log("  ⚠ lib3mf não disponível, usando trimesh")
        
        # Carregar dados
        self.log("  Carregando dados...")
        data = np.load(npz_path)
        vertices = data['vertices']
        faces = data['faces']
        palette = data['palette']
        
        # Detectar formato (novo ou antigo)
        if 'face_palette_indices' in data:
            # NOVO: índices diretos (super rápido)
            colors = data['face_palette_indices']
            self.log(f"  {len(vertices)} vértices, {len(faces)} faces [modo rápido]")
        elif 'colors' in data:
            # ANTIGO: cores RGB
            colors = data['colors']
            self.log(f"  {len(vertices)} vértices, {len(faces)} faces [modo compatibilidade]")
        else:
            raise RuntimeError("Formato de dados inválido")
        
        if use_lib3mf:
            self.build_with_lib3mf(vertices, faces, colors, palette, out_3mf)
        else:
            self.build_with_trimesh(vertices, faces, colors, palette, out_3mf)
    
    def build_with_lib3mf(self, vertices, faces, colors, palette, out_3mf):
        """Constrói 3MF com lib3mf - cores aplicadas durante criação"""
        import Lib3MF
        import numpy as np
        
        self.log("  Criando modelo 3MF...")
        wrapper = Lib3MF.Wrapper()
        model = wrapper.CreateModel()
        
        # Criar materiais da paleta
        self.log("  Criando materiais...")
        base_materials = model.AddBaseMaterialGroup()
        
        palette_int = (palette * 255).astype(np.uint8)
        for i, color in enumerate(palette_int):
            base_materials.AddMaterial(
                f"Color{i}",
                Lib3MF.sColor(int(color[0]), int(color[1]), int(color[2]), 255)
            )
        
        # Mapear cores RGB para índices de material
        self.log("  Mapeando cores para materiais...")
        color_to_idx = {}
        for i, p_color in enumerate(palette):
            key = tuple((p_color * 255).astype(np.uint8))
            color_to_idx[key] = i
        
        # Criar mesh object
        mesh_obj = model.AddMeshObject()
        mesh_obj.SetName("colored_model")
        
        # Adicionar vértices
        self.log("  Adicionando vértices...")
        for v in vertices:
            mesh_obj.AddVertex(Lib3MF.sPosition(float(v[0]), float(v[1]), float(v[2])))
        
        # Adicionar triângulos COM CORES simultaneamente
        self.log("  Adicionando triângulos com cores...")
        colors_int = (colors * 255).astype(np.uint8)
        
        batch_log = 25000
        for i, (face, face_colors) in enumerate(zip(faces, colors_int)):
            if i % batch_log == 0 and i > 0:
                self.log(f"    {i}/{len(faces)} ({(i/len(faces)*100):.1f}%)")
            
            # Adicionar triângulo
            tri = Lib3MF.sTriangle(int(face[0]), int(face[1]), int(face[2]))
            mesh_obj.AddTriangle(tri)
            
            # Mapear cores dos 3 vértices para índices de material
            mat_ids = []
            for vert_color in face_colors:
                key = tuple(vert_color)
                mat_id = color_to_idx.get(key, 0)
                mat_ids.append(mat_id)
            
            # Aplicar propriedades (cores) ao triângulo
            mesh_obj.SetTriangleProperties(i, base_materials, mat_ids[0], mat_ids[1], mat_ids[2])
        
        self.log("  Finalizando 3MF...")
        model.AddBuildItem(mesh_obj, wrapper.GetIdentityTransform())
        writer = model.QueryWriter("3mf")
        writer.WriteToFile(out_3mf)
        self.log("  ✓ 3MF criado com sucesso")
    
    def build_with_trimesh(self, vertices, faces, colors, palette, out_3mf):
        """Fallback com trimesh (pode perder cores)"""
        import trimesh
        import numpy as np
        
        self.log("  ⚠ Construindo com trimesh - cores podem não funcionar no slicer")
        
        # Converter cores para vertex colors
        vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        vertex_colors[:, 3] = 255  # Alpha
        
        if colors.shape == (len(faces), 3) and colors.dtype in [np.int32, np.int64]:
            # Índices de paleta
            for face_idx, (face, color_indices) in enumerate(zip(faces, colors)):
                for vert_idx, pal_idx in zip(face, color_indices):
                    vertex_colors[vert_idx, :3] = (palette[pal_idx] * 255).astype(np.uint8)
        else:
            # Cores RGB
            for face_idx, (face, face_colors) in enumerate(zip(faces, colors)):
                for vert_idx, vert_color in zip(face, face_colors):
                    vertex_colors[vert_idx, :3] = (vert_color * 255).astype(np.uint8)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        mesh.export(out_3mf, file_type='3mf')
        self.log("  ✓ 3MF criado (sem garantia de cores)")
    
    def log(self, msg):
        self.log_signal.emit(f"[{time.strftime('%H:%M:%S')}] {msg}")

class PreviewThread(QtCore.QThread):
    preview_ready = QtCore.pyqtSignal(str)
    log_signal = QtCore.pyqtSignal(str)
    
    def __init__(self, blender_path, fbx_path, out_img, res_x=512, res_y=512, parent=None):
        super().__init__(parent)
        self.blender_path = blender_path
        self.fbx_path = fbx_path
        self.out_img = out_img
        self.res_x = res_x
        self.res_y = res_y
    
    def run(self):
        tmpdir = None
        try:
            script = BLENDER_PREVIEW_SCRIPT.format(
                fbx_path=self.fbx_path.replace("\\","/"),
                out_path=self.out_img.replace("\\","/"),
                res_x=self.res_x, res_y=self.res_y
            )
            tmpdir = tempfile.mkdtemp(prefix="fbx_preview_")
            script_file = os.path.join(tmpdir, "preview.py")
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script)
            
            cmd = [self.blender_path, "--background", "--python", script_file]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            
            rc, out, err = run_process_stream(cmd, timeout=60, creationflags=creationflags)
            
            if rc == 0 and os.path.exists(self.out_img):
                self.preview_ready.emit(self.out_img)
            else:
                self.log_signal.emit(f"Preview falhou (rc={rc})")
        except Exception as e:
            self.log_signal.emit(f"Preview erro: {e}")
        finally:
            if tmpdir:
                try:
                    shutil.rmtree(tmpdir)
                except:
                    pass

# ========== Main Window ==========
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FBX → 3MF Colorido - Otimizado")
        self.resize(1200, 800)
        
        self.central = QtWidgets.QWidget()
        self.setCentralWidget(self.central)
        self.layout = QtWidgets.QHBoxLayout(self.central)
        
        # ===== LEFT PANEL =====
        self.left = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QVBoxLayout(self.left)
        self.layout.addWidget(self.left, 2)
        
        title = QtWidgets.QLabel("Conversor FBX → 3MF")
        font = title.font()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        self.left_layout.addWidget(title)
        
        # Blender path
        hb = QtWidgets.QHBoxLayout()
        self.blender_line = QtWidgets.QLineEdit()
        self.blender_btn = QtWidgets.QPushButton("Selecionar Blender")
        hb.addWidget(self.blender_line)
        hb.addWidget(self.blender_btn)
        self.left_layout.addLayout(hb)
        self.blender_btn.clicked.connect(self.select_blender)
        
        # FBX input
        hb2 = QtWidgets.QHBoxLayout()
        self.fbx_line = QtWidgets.QLineEdit()
        self.fbx_btn = QtWidgets.QPushButton("Abrir FBX")
        hb2.addWidget(self.fbx_line)
        hb2.addWidget(self.fbx_btn)
        self.left_layout.addLayout(hb2)
        self.fbx_btn.clicked.connect(self.select_fbx)
        
        # Output
        hb3 = QtWidgets.QHBoxLayout()
        self.out_line = QtWidgets.QLineEdit()
        self.out_btn = QtWidgets.QPushButton("Salvar 3MF")
        hb3.addWidget(self.out_line)
        hb3.addWidget(self.out_btn)
        self.left_layout.addLayout(hb3)
        self.out_btn.clicked.connect(self.select_out)
        
        # Options
        opt_group = QtWidgets.QGroupBox("Opções")
        opt_layout = QtWidgets.QVBoxLayout()
        
        hb_colors = QtWidgets.QHBoxLayout()
        hb_colors.addWidget(QtWidgets.QLabel("Máx. cores:"))
        self.spin_colors = QtWidgets.QSpinBox()
        self.spin_colors.setRange(2, 16)
        self.spin_colors.setValue(8)
        hb_colors.addWidget(self.spin_colors)
        opt_layout.addLayout(hb_colors)
        
        opt_group.setLayout(opt_layout)
        self.left_layout.addWidget(opt_group)
        
        # Convert button
        self.convert_btn = QtWidgets.QPushButton("CONVERTER")
        self.convert_btn.setEnabled(False)
        self.convert_btn.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        self.left_layout.addWidget(self.convert_btn)
        self.convert_btn.clicked.connect(self.start_conversion)
        
        # Progress
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.left_layout.addWidget(self.progress)
        
        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(200)
        self.left_layout.addWidget(self.log)
        
        # ===== RIGHT PANEL =====
        self.right = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout(self.right)
        self.layout.addWidget(self.right, 1)
        
        # Preview
        preview_group = QtWidgets.QGroupBox("Preview do Modelo")
        preview_layout = QtWidgets.QVBoxLayout()
        
        self.preview_label = QtWidgets.QLabel("Preview")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedSize(380, 380)
        self.preview_label.setStyleSheet("background:#ddd; border:1px solid #aaa;")
        preview_layout.addWidget(self.preview_label)
        
        self.preview_btn = QtWidgets.QPushButton("Gerar Preview")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self.generate_preview)
        preview_layout.addWidget(self.preview_btn)
        
        preview_group.setLayout(preview_layout)
        self.right_layout.addWidget(preview_group)
        
        # Palette widget
        self.palette_widget = PaletteWidget()
        self.palette_widget.palette_changed.connect(self.on_palette_changed)
        self.right_layout.addWidget(self.palette_widget)
        
        # State
        self.blender_path = ""
        self.fbx_path = ""
        self.out_path = ""
        self.current_palette = []
        
        self.auto_find_blender()
    
    def auto_find_blender(self):
        candidates = []
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        
        for base in [pf, pf_x86]:
            bf = os.path.join(base, "Blender Foundation")
            if os.path.exists(bf):
                for nm in os.listdir(bf):
                    cand = os.path.join(bf, nm, "blender.exe")
                    if os.path.exists(cand):
                        candidates.append(cand)
        
        import shutil
        found = shutil.which("blender")
        if found:
            candidates.append(found)
        
        if candidates:
            self.blender_path = candidates[0]
            self.blender_line.setText(self.blender_path)
            self.log_append(f"Blender detectado: {Path(self.blender_path).name}")
    
    def select_blender(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecione Blender", "", "Blender (blender*);;Todos (*)")
        if path:
            self.blender_path = path
            self.blender_line.setText(path)
            self.log_append(f"Blender: {Path(path).name}")
            self.update_buttons()
    
    def select_fbx(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Selecione FBX", "", "FBX (*.fbx);;Todos (*)")
        if path:
            self.fbx_path = path
            self.fbx_line.setText(path)
            self.log_append(f"FBX: {Path(path).name}")
            
            default_out = str(Path(path).with_suffix('.3mf'))
            self.out_line.setText(default_out)
            self.out_path = default_out
            
            self.preview_btn.setEnabled(bool(self.blender_path))
            self.update_buttons()
            
            # Auto-extrair cores
            if self.blender_path:
                self.extract_colors()
    
    def select_out(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Salvar 3MF", self.out_line.text(), "3MF (*.3mf);;Todos (*)")
        if path:
            self.out_path = path
            self.out_line.setText(path)
            self.log_append(f"Saída: {Path(path).name}")
            self.update_buttons()
    
    def update_buttons(self):
        has_palette = len(self.current_palette) > 0
        enabled = bool(self.blender_path and self.fbx_path and self.out_path and has_palette)
        self.convert_btn.setEnabled(enabled)
        self.preview_btn.setEnabled(bool(self.blender_path and self.fbx_path))
    
    def log_append(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {text}")
    
    def extract_colors(self):
        self.log_append("Detectando cores...")
        self.progress.setVisible(True)
        max_colors = self.spin_colors.value()
        
        self.extract_thread = ColorExtractThread(self.blender_path, self.fbx_path, max_colors)
        self.extract_thread.log_signal.connect(self.log_append)
        self.extract_thread.colors_ready.connect(self.on_colors_extracted)
        self.extract_thread.start()
    
    def on_colors_extracted(self, colors):
        self.progress.setVisible(False)
        self.current_palette = colors
        self.palette_widget.set_palette(colors)
        self.log_append(f"✓ {len(colors)} cores detectadas")
        self.update_buttons()
    
    def on_palette_changed(self):
        self.current_palette = self.palette_widget.get_palette()
        self.log_append(f"Paleta atualizada: {len(self.current_palette)} cores")
    
    def start_conversion(self):
        if not self.current_palette:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Detecte as cores primeiro!")
            return
        
        self.progress.setVisible(True)
        self.convert_btn.setEnabled(False)
        
        palette = self.palette_widget.get_palette()
        self.convert_thread = ConvertThread(self.blender_path, self.fbx_path, self.out_path, palette)
        self.convert_thread.log_signal.connect(self.log_append)
        self.convert_thread.finished_signal.connect(self.on_conversion_finished)
        self.convert_thread.start()
    
    def on_conversion_finished(self, success, message):
        self.progress.setVisible(False)
        self.convert_btn.setEnabled(True)
        if success:
            QtWidgets.QMessageBox.information(self, "Pronto!", message)
        else:
            QtWidgets.QMessageBox.critical(self, "Erro", message)
    
    def generate_preview(self):
        tmp = tempfile.gettempdir()
        out_img = os.path.join(tmp, f"fbx_preview_{int(time.time())}.png")
        
        self.preview_thread = PreviewThread(self.blender_path, self.fbx_path, out_img, 512, 512)
        self.preview_thread.log_signal.connect(self.log_append)
        self.preview_thread.preview_ready.connect(self.on_preview_ready)
        self.preview_btn.setEnabled(False)
        self.log_append("Gerando preview...")
        self.preview_thread.start()
    
    def on_preview_ready(self, path):
        try:
            pix = QtGui.QPixmap(path)
            pix = pix.scaled(self.preview_label.size(), 
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                        QtCore.Qt.TransformationMode.SmoothTransformation)
            self.preview_label.setPixmap(pix)
            self.preview_label.setText("")
            self.log_append("✓ Preview pronto")
        except Exception as e:
            self.log_append(f"Erro no preview: {e}")
        finally:
            self.preview_btn.setEnabled(True)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
