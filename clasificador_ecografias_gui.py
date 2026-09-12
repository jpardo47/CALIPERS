#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Clasificador Gráfico de Ecografías Mamarias para Fine-Tuning de LaMa
=============================================================================
Herramienta interactiva para separar rápidamente ecografías limpias
(sin calipers, texto ni marcas) destinadas a Google Drive / Google Colab.

Teclas rápidas (Hotkeys):
  [ ← ] / [ A ] / [ 1 ] : Sin anotaciones, texto y demás (Limpia -> Copia a Drive)
  [ → ] / [ D ] / [ 2 ] : Con anotaciones, texto y demás (Marcada)
  [ Z ] / [ Ctrl+Z ]    : Deshacer última acción (Undo)
  [ Espacio ] / [ S ]   : Saltar imagen sin clasificar
=============================================================================
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# Directorios por defecto relativos al proyecto
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent / "data" / "bases_datos_originales"
DEFAULT_TARGET_DIR = Path(__file__).resolve().parent / "ecografias_limpias_para_drive"
LOG_FILENAME = "clasificador_historial.json"


class UltrasoundClassifierApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🩺 Curador de Ecografías Mamarias | Preparación para LaMa")
        self.root.geometry("1100x820")
        self.root.minsize(850, 650)

        # Configurar colores del tema moderno
        self.bg_color = "#18181b"          # Fondo oscuro elegante
        self.card_bg = "#27272a"           # Fondo de tarjetas
        self.fg_color = "#f4f4f5"          # Texto blanco suave
        self.accent_green = "#10b981"      # Verde esmeralda (Limpia)
        self.accent_green_hover = "#059669"
        self.accent_red = "#ef4444"        # Rojo coral (Anotada)
        self.accent_red_hover = "#dc2626"
        self.accent_blue = "#3b82f6"       # Azul acción
        self.border_color = "#3f3f46"

        self.root.configure(bg=self.bg_color)

        # Rutas de trabajo
        self.source_dir = Path(DEFAULT_SOURCE_DIR if DEFAULT_SOURCE_DIR.exists() else Path.cwd())
        self.target_dir = Path(DEFAULT_TARGET_DIR)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.target_dir / LOG_FILENAME

        # Estado interno
        self.image_files = []
        self.current_idx = 0
        self.history = []          # Pila para Deshacer: [(source_path, dest_path_or_None, decision)]
        self.history_records = {}  # {str(filepath): 'clean' | 'annotated' | 'skipped'}
        self.current_image_pil = None
        self.current_tk_image = None

        # Cargar registro previo si existe para reanudar sesión
        self._load_session_history()

        # Construir Interfaz
        self._build_ui()

        # Escanear imágenes de origen
        self._scan_images()

        # Enlazar atajos de teclado
        self._bind_shortcuts()

        # Mostrar primera imagen
        self._show_current_image()

    def _load_session_history(self):
        """Carga el historial previo para no volver a evaluar imágenes ya clasificadas."""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    self.history_records = json.load(f)
            except Exception as e:
                print(f"No se pudo cargar el historial previo: {e}")
                self.history_records = {}

    def _save_session_history(self):
        """Guarda el historial de clasificación en disco."""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.history_records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error al guardar historial: {e}")

    def _build_ui(self):
        # 1. Barra Superior (Información y Controles de Rutas)
        top_bar = tk.Frame(self.root, bg=self.card_bg, padx=16, pady=10)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            top_bar,
            text="Curador de Ecografías: Filtro para Fine-Tuning de LaMa",
            font=("Segoe UI", 13, "bold"),
            fg=self.fg_color,
            bg=self.card_bg,
        )
        title_lbl.pack(side=tk.LEFT)

        # Botones de utilidad superior
        btn_zip = tk.Button(
            top_bar,
            text="📦 Comprimir en .ZIP para Drive",
            font=("Segoe UI", 9, "bold"),
            bg="#0284c7",
            fg="white",
            activebackground="#0369a1",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._compress_to_zip,
        )
        btn_zip.pack(side=tk.RIGHT, padx=6)

        btn_open_folder = tk.Button(
            top_bar,
            text="📂 Ver Limpias",
            font=("Segoe UI", 9),
            bg="#3f3f46",
            fg="white",
            activebackground="#52525b",
            activeforeground="white",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2",
            command=self._open_target_folder,
        )
        btn_open_folder.pack(side=tk.RIGHT, padx=6)

        btn_change_source = tk.Button(
            top_bar,
            text="📁 Cambiar Carpeta Origen",
            font=("Segoe UI", 9),
            bg="#3f3f46",
            fg="white",
            activebackground="#52525b",
            activeforeground="white",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2",
            command=self._change_source_folder,
        )
        btn_change_source.pack(side=tk.RIGHT, padx=6)

        # 2. Barra de Estado / Métricas
        metrics_bar = tk.Frame(self.root, bg=self.bg_color, padx=16, pady=8)
        metrics_bar.pack(side=tk.TOP, fill=tk.X)

        self.info_lbl = tk.Label(
            metrics_bar,
            text="Cargando dataset...",
            font=("Segoe UI", 10),
            fg="#a1a1aa",
            bg=self.bg_color,
        )
        self.info_lbl.pack(side=tk.LEFT)

        self.counter_lbl = tk.Label(
            metrics_bar,
            text="Limpias guardadas: 0 | Con anotaciones: 0",
            font=("Segoe UI", 10, "bold"),
            fg="#38bdf8",
            bg=self.bg_color,
        )
        self.counter_lbl.pack(side=tk.RIGHT)

        # Barra de progreso
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self.root,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.progress_bar.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(0, 6))

        # 3. Contenedor Central de Imagen
        self.image_container = tk.Frame(self.root, bg="#09090b", relief=tk.SUNKEN, bd=1)
        self.image_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=6)
        self.image_container.bind("<Configure>", self._on_resize)

        self.image_canvas = tk.Label(self.image_container, bg="#09090b")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        # Nombre del archivo actual
        self.filename_lbl = tk.Label(
            self.root,
            text="Archivo: -",
            font=("Consolas", 10),
            fg="#e4e4e7",
            bg=self.bg_color,
            pady=4,
        )
        self.filename_lbl.pack(side=tk.TOP)

        # 4. Panel Inferior de Botones Principales
        bottom_panel = tk.Frame(self.root, bg=self.card_bg, padx=16, pady=16)
        bottom_panel.pack(side=tk.BOTTOM, fill=tk.X)

        bottom_panel.columnconfigure(0, weight=4)  # Botón Limpia
        bottom_panel.columnconfigure(1, weight=1)  # Botón Deshacer
        bottom_panel.columnconfigure(2, weight=1)  # Botón Saltar
        bottom_panel.columnconfigure(3, weight=4)  # Botón Con Anotaciones

        # Botón 1: SIN ANOTACIONES (Limpia)
        self.btn_clean = tk.Button(
            bottom_panel,
            text="✅  SIN ANOTACIONES, TEXTO Y DEMÁS\n[ Tecla A / ← / 1 ]  (Guardar para Drive)",
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_green,
            fg="white",
            activebackground=self.accent_green_hover,
            activeforeground="white",
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=self._mark_as_clean,
        )
        self.btn_clean.grid(row=0, column=0, sticky="nsew", padx=6)

        # Botón Deshacer
        self.btn_undo = tk.Button(
            bottom_panel,
            text="↩ Deshacer\n[ Ctrl+Z / Z ]",
            font=("Segoe UI", 9, "bold"),
            bg="#4b5563",
            fg="white",
            activebackground="#374151",
            activeforeground="white",
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=self._undo_last_action,
        )
        self.btn_undo.grid(row=0, column=1, sticky="nsew", padx=4)

        # Botón Saltar
        self.btn_skip = tk.Button(
            bottom_panel,
            text="⏭ Saltar\n[ Espacio ]",
            font=("Segoe UI", 9),
            bg="#3f3f46",
            fg="white",
            activebackground="#52525b",
            activeforeground="white",
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=self._skip_image,
        )
        self.btn_skip.grid(row=0, column=2, sticky="nsew", padx=4)

        # Botón 2: CON ANOTACIONES (Marcada)
        self.btn_annotated = tk.Button(
            bottom_panel,
            text="❌  CON ANOTACIONES, TEXTO Y DEMÁS\n[ Tecla D / → / 2 ]  (Descartar de limpias)",
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_red,
            fg="white",
            activebackground=self.accent_red_hover,
            activeforeground="white",
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=self._mark_as_annotated,
        )
        self.btn_annotated.grid(row=0, column=3, sticky="nsew", padx=6)

    def _bind_shortcuts(self):
        """Asocia las teclas rápidas para clasificar a máxima velocidad."""
        # Limpia
        self.root.bind("<a>", lambda e: self._mark_as_clean())
        self.root.bind("<A>", lambda e: self._mark_as_clean())
        self.root.bind("<Left>", lambda e: self._mark_as_clean())
        self.root.bind("<KP_1>", lambda e: self._mark_as_clean())
        self.root.bind("1", lambda e: self._mark_as_clean())

        # Con anotaciones
        self.root.bind("<d>", lambda e: self._mark_as_annotated())
        self.root.bind("<D>", lambda e: self._mark_as_annotated())
        self.root.bind("<Right>", lambda e: self._mark_as_annotated())
        self.root.bind("<KP_2>", lambda e: self._mark_as_annotated())
        self.root.bind("2", lambda e: self._mark_as_annotated())

        # Deshacer y saltar
        self.root.bind("<z>", lambda e: self._undo_last_action())
        self.root.bind("<Z>", lambda e: self._undo_last_action())
        self.root.bind("<Control-z>", lambda e: self._undo_last_action())
        self.root.bind("<Control-Z>", lambda e: self._undo_last_action())
        self.root.bind("<space>", lambda e: self._skip_image())
        self.root.bind("<s>", lambda e: self._skip_image())
        self.root.bind("<S>", lambda e: self._skip_image())

    def _scan_images(self):
        """Busca recursivamente imágenes en la carpeta fuente excluyendo las ya evaluadas si se desea."""
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        if not self.source_dir.exists():
            messagebox.showwarning("Aviso", f"La carpeta de origen no existe:\n{self.source_dir}")
            return

        all_paths = []
        for p in self.source_dir.rglob("*"):
            if p.suffix.lower() in valid_extensions and not p.name.startswith("."):
                # Evitar incluir imágenes de la propia carpeta de destino
                if self.target_dir not in p.parents:
                    all_paths.append(p)

        all_paths.sort()
        self.image_files = all_paths

        # Avanzar hasta la primera imagen pendiente
        self.current_idx = 0
        while self.current_idx < len(self.image_files):
            key = str(self.image_files[self.current_idx])
            if key not in self.history_records:
                break
            self.current_idx += 1

        self._update_counters()

    def _update_counters(self):
        total = len(self.image_files)
        clean_count = sum(1 for v in self.history_records.values() if v == "clean")
        annotated_count = sum(1 for v in self.history_records.values() if v == "annotated")
        processed_count = len(self.history_records)

        progress_pct = (processed_count / total * 100.0) if total > 0 else 0.0
        self.progress_var.set(progress_pct)

        curr_num = min(self.current_idx + 1, total)
        self.info_lbl.config(
            text=f"Progreso: {curr_num} de {total} ({progress_pct:.1f}%) | Pendientes: {max(0, total - processed_count)}"
        )
        self.counter_lbl.config(
            text=f"✨ Limpias: {clean_count}  |  🏷️ Con anotaciones: {annotated_count}"
        )

    def _show_current_image(self):
        if not self.image_files or self.current_idx >= len(self.image_files):
            self.filename_lbl.config(text="🎉 ¡Todas las imágenes han sido evaluadas!")
            self.image_canvas.config(image="", text="¡Completado!\nPuedes hacer clic en 'Comprimir en .ZIP para Drive'.", fg="white", font=("Segoe UI", 16))
            return

        img_path = self.image_files[self.current_idx]
        dataset_name = img_path.parent.name
        self.filename_lbl.config(text=f"[{dataset_name}] {img_path.name}  ({img_path.stat().st_size // 1024} KB)")

        try:
            self.current_image_pil = Image.open(img_path)
            self._render_image()
        except Exception as e:
            self.filename_lbl.config(text=f"Error cargando imagen: {e}")
            self.image_canvas.config(image="", text="Error al leer imagen")

    def _render_image(self):
        if self.current_image_pil is None:
            return

        # Dimensiones del contenedor
        container_w = max(self.image_container.winfo_width() - 20, 300)
        container_h = max(self.image_container.winfo_height() - 20, 300)

        # Calcular aspect ratio
        img_w, img_h = self.current_image_pil.size
        ratio = min(container_w / img_w, container_h / img_h)
        new_w = max(1, int(img_w * ratio))
        new_h = max(1, int(img_h * ratio))

        resized = self.current_image_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
        self.current_tk_image = ImageTk.PhotoImage(resized)

        self.image_canvas.config(image=self.current_tk_image, text="")

    def _on_resize(self, event):
        """Redimensiona suavemente la imagen cuando el usuario cambia el tamaño de la ventana."""
        self._render_image()

    def _mark_as_clean(self):
        """Acción: La imagen no tiene artefactos. Se copia a la carpeta de limpias para Drive."""
        if self.current_idx >= len(self.image_files):
            return

        src_path = self.image_files[self.current_idx]
        
        # Generar nombre único para evitar colisiones entre datasets
        dataset_prefix = src_path.parent.name
        dest_filename = f"{dataset_prefix}_{src_path.name}"
        dest_path = self.target_dir / dest_filename

        try:
            shutil.copy2(src_path, dest_path)
            self.history_records[str(src_path)] = "clean"
            self.history.append((src_path, dest_path, "clean"))
            self._save_session_history()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo copiar la imagen:\n{e}")
            return

        self._advance_next()

    def _mark_as_annotated(self):
        """Acción: La imagen tiene calipers o texto. Se registra como anotada."""
        if self.current_idx >= len(self.image_files):
            return

        src_path = self.image_files[self.current_idx]
        self.history_records[str(src_path)] = "annotated"
        self.history.append((src_path, None, "annotated"))
        self._save_session_history()

        self._advance_next()

    def _skip_image(self):
        """Salta la imagen sin registrarla."""
        if self.current_idx >= len(self.image_files):
            return

        src_path = self.image_files[self.current_idx]
        self.history.append((src_path, None, "skipped"))
        self.current_idx += 1
        self._update_counters()
        self._show_current_image()

    def _advance_next(self):
        self.current_idx += 1
        self._update_counters()
        self._show_current_image()

    def _undo_last_action(self):
        """Deshace la última decisión y elimina el archivo copiado si era limpia."""
        if not self.history:
            messagebox.showinfo("Deshacer", "No hay acciones previas para deshacer.")
            return

        src_path, dest_path, decision = self.history.pop()

        # Si se había copiado a limpias, eliminar la copia
        if decision == "clean" and dest_path and dest_path.exists():
            try:
                os.remove(dest_path)
            except Exception as e:
                print(f"Error eliminando copia al deshacer: {e}")

        # Retirar del registro
        if str(src_path) in self.history_records:
            del self.history_records[str(src_path)]
            self._save_session_history()

        # Retroceder índice
        if src_path in self.image_files:
            self.current_idx = self.image_files.index(src_path)
        else:
            self.current_idx = max(0, self.current_idx - 1)

        self._update_counters()
        self._show_current_image()

    def _compress_to_zip(self):
        """Empaqueta toda la carpeta de limpias en un archivo .zip listo para Drive."""
        clean_files = [p for p in self.target_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        if not clean_files:
            messagebox.showinfo("Aviso", "Aún no hay imágenes limpias clasificadas para comprimir.")
            return

        zip_out = self.target_dir.parent / "ecografias_limpias_para_drive.zip"
        
        try:
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
                for img in clean_files:
                    zf.write(img, arcname=f"ecografias_limpias/{img.name}")
            
            size_mb = zip_out.stat().st_size / (1024 * 1024)
            messagebox.showinfo(
                "¡Comprimido con Éxito!",
                f"Se ha creado el archivo comprimido listo para Google Drive:\n\n"
                f"📁 {zip_out.name}\n"
                f"📊 Total imágenes: {len(clean_files)}\n"
                f"💾 Tamaño: {size_mb:.2f} MB\n\n"
                f"Ruta: {zip_out}"
            )
        except Exception as e:
            messagebox.showerror("Error de compresión", f"No se pudo crear el archivo zip:\n{e}")

    def _open_target_folder(self):
        """Abre el explorador de archivos de Windows en la carpeta de limpias."""
        try:
            if sys.platform == "win32":
                os.startfile(str(self.target_dir))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(self.target_dir)])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la carpeta:\n{e}")

    def _change_source_folder(self):
        """Permite al usuario seleccionar otra carpeta que contenga ecografías."""
        selected = filedialog.askdirectory(
            title="Seleccionar carpeta con ecografías",
            initialdir=str(self.source_dir),
        )
        if selected:
            self.source_dir = Path(selected)
            self._scan_images()
            self._show_current_image()


def main():
    root = tk.Tk()
    app = UltrasoundClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
