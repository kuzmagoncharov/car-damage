"""Окно: загрузить или вставить фото автомобиля и получить результат."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageGrab, ImageTk

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import CarDamageModel
from src.config import IMAGE_EXTENSIONS, configure_console

VERDICTS = {
    "damaged": "Автомобиль повреждён",
    "normal": "Повреждений не видно",
}
COLORS = {
    "damaged": "#b42318",
    "normal": "#067647",
}


class CarDamageApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Анализ аварийности автомобиля")
        self.root.geometry("560x740")
        self.root.minsize(500, 680)
        self.root.configure(bg="#f4f4f5")

        self.model: CarDamageModel | None = None
        self.current_image: Image.Image | None = None
        self.preview_image: ImageTk.PhotoImage | None = None

        self._build()
        self.root.bind("<Control-v>", self._on_paste)
        self.root.bind("<Control-V>", self._on_paste)
        self.root.after(80, self._load_model)

    def _build(self) -> None:
        title = tk.Label(
            self.root,
            text="Анализ аварийности автомобиля",
            font=("Segoe UI", 16, "bold"),
            bg="#f4f4f5",
            fg="#18181b",
        )
        title.pack(pady=(20, 4))

        self.status = tk.Label(
            self.root,
            text="Загрузка модели…",
            font=("Segoe UI", 10),
            bg="#f4f4f5",
            fg="#52525b",
        )
        self.status.pack()

        self.preview = tk.Label(
            self.root,
            text="Нажмите, чтобы выбрать фото\nили Ctrl+V — вставить из буфера",
            font=("Segoe UI", 11),
            bg="#e4e4e7",
            fg="#71717a",
            width=56,
            height=16,
            relief="flat",
            cursor="hand2",
        )
        self.preview.pack(fill="both", expand=True, padx=20, pady=12)
        self.preview.bind("<Button-1>", self._on_preview_click)

        buttons = tk.Frame(self.root, bg="#f4f4f5")
        buttons.pack(pady=4)

        self.pick_btn = ttk.Button(
            buttons,
            text="Выбрать фото",
            command=self._pick_photo,
            state="disabled",
        )
        self.pick_btn.pack(side="left", padx=6)

        self.paste_btn = ttk.Button(
            buttons,
            text="Вставить  Ctrl+V",
            command=self._paste_photo,
            state="disabled",
        )
        self.paste_btn.pack(side="left", padx=6)

        self.check_btn = ttk.Button(
            buttons,
            text="Проверить ещё раз",
            command=self._run_predict,
            state="disabled",
        )
        self.check_btn.pack(side="left", padx=6)

        self.verdict = tk.Label(
            self.root,
            text="Вердикт: —",
            font=("Segoe UI", 15, "bold"),
            bg="#f4f4f5",
            fg="#18181b",
        )
        self.verdict.pack(padx=20, pady=(12, 10))

        meters = tk.Frame(self.root, bg="#f4f4f5")
        meters.pack(fill="x", padx=36, pady=(0, 18))

        self.emergency_bar, self.emergency_value = self._add_meter(
            meters,
            "Аварийность",
            0,
            COLORS["damaged"],
        )
        self.integrity_bar, self.integrity_value = self._add_meter(
            meters,
            "Целостность",
            1,
            COLORS["normal"],
        )

    def _add_meter(
        self,
        parent: tk.Frame,
        title: str,
        row: int,
        color: str,
    ) -> tuple[tk.Frame, tk.Label]:
        tk.Label(
            parent,
            text=title,
            font=("Segoe UI", 10),
            bg="#f4f4f5",
            fg="#3f3f46",
            anchor="w",
        ).grid(row=row * 2, column=0, sticky="w", pady=(8, 2))

        trough = tk.Frame(parent, bg="#e4e4e7", height=16)
        trough.grid(row=row * 2 + 1, column=0, sticky="ew", padx=(0, 10))
        trough.grid_propagate(False)
        fill = tk.Frame(trough, bg=color, height=16)
        fill.place(x=0, y=0, relheight=1, relwidth=0)
        trough.fill = fill  # type: ignore[attr-defined]

        value = tk.Label(
            parent,
            text="—",
            font=("Segoe UI", 11, "bold"),
            bg="#f4f4f5",
            fg=color,
            width=5,
            anchor="e",
        )
        value.grid(row=row * 2 + 1, column=1, sticky="e")
        parent.columnconfigure(0, weight=1)
        return trough, value

    def _set_bar(self, bar: tk.Frame, percent: float) -> None:
        bar.fill.place(x=0, y=0, relheight=1, relwidth=max(0.0, min(1.0, percent)))  # type: ignore[attr-defined]

    def _load_model(self) -> None:
        try:
            self.model = CarDamageModel()
        except FileNotFoundError as exc:
            self.status.config(text=str(exc), fg="#b42318")
            messagebox.showerror("Модель не найдена", str(exc))
            return
        except Exception as exc:
            self.status.config(text=f"Не удалось загрузить модель: {exc}", fg="#b42318")
            messagebox.showerror("Ошибка", str(exc))
            return

        self.status.config(
            text="Готово: нажмите на область фото или вставьте Ctrl+V",
            fg="#067647",
        )
        self.pick_btn.config(state="normal")
        self.paste_btn.config(state="normal")

    def _on_preview_click(self, _event: tk.Event) -> None:
        if self.model is None:
            return
        self._pick_photo()

    def _on_paste(self, _event: tk.Event) -> str:
        self._paste_photo()
        return "break"

    def _pick_photo(self) -> None:
        if self.model is None:
            return
        filetypes = [
            ("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp"),
            ("Все файлы", "*.*"),
        ]
        chosen = filedialog.askopenfilename(title="Выберите фото автомобиля", filetypes=filetypes)
        if not chosen:
            return

        path = Path(chosen)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            messagebox.showwarning("Формат", "Нужно фото: PNG, JPG, JPEG, BMP или WEBP.")
            return

        self._set_image(Image.open(path), source=path.name)

    def _paste_photo(self) -> None:
        if self.model is None:
            return

        image = self._image_from_clipboard()
        if image is None:
            messagebox.showinfo(
                "Буфер обмена",
                "В буфере нет фото.\nСкопируйте снимок и нажмите Ctrl+V.",
            )
            return
        self._set_image(image, source="вставка из буфера")

    def _image_from_clipboard(self) -> Image.Image | None:
        clip = ImageGrab.grabclipboard()
        if isinstance(clip, Image.Image):
            return clip
        if isinstance(clip, list):
            for item in clip:
                path = Path(item)
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    return Image.open(path)
        return None

    def _set_image(self, image: Image.Image, source: str) -> None:
        rgb = image.convert("RGB")
        self.current_image = rgb
        self._show_preview(rgb)
        self.check_btn.config(state="normal")
        self.status.config(text=source, fg="#52525b")
        self._clear_result()
        self._run_predict()

    def _show_preview(self, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((480, 320))
        self.preview_image = ImageTk.PhotoImage(preview)
        self.preview.config(image=self.preview_image, text="", width=480, height=320)

    def _clear_result(self) -> None:
        self.verdict.config(text="Вердикт: —", fg="#18181b")
        self._set_bar(self.emergency_bar, 0)
        self._set_bar(self.integrity_bar, 0)
        self.emergency_value.config(text="—")
        self.integrity_value.config(text="—")

    def _run_predict(self) -> None:
        if self.model is None or self.current_image is None:
            return

        self.status.config(text="Проверяю фото…", fg="#52525b")
        self.root.update_idletasks()
        try:
            label, _confidence, scores = self.model.predict_image(self.current_image)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
            self.status.config(text="Не получилось проверить фото.", fg="#b42318")
            return

        emergency = scores.get("damaged", 0.0)
        integrity = scores.get("normal", 0.0)
        color = COLORS.get(label, "#18181b")
        self.verdict.config(text=VERDICTS.get(label, label), fg=color)
        self._set_bar(self.emergency_bar, emergency)
        self._set_bar(self.integrity_bar, integrity)
        self.emergency_value.config(text=f"{emergency:.0%}")
        self.integrity_value.config(text=f"{integrity:.0%}")
        self.status.config(text="Готово", fg="#067647")


def main() -> None:
    configure_console()
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    CarDamageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
