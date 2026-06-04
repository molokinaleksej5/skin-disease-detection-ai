import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import torch

from src.config import CFG
from src.inference.predict import predict_image


def get_device_text() -> str:
    # Порядок важен: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return "Устройство: CUDA (GPU)"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Устройство: MPS (Apple Silicon)"
    return "Устройство: CPU"


def load_label_map() -> dict:
    with open(CFG.label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


class MedicalNNApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MedicalNN - Диагностика кожных поражений")
        self.geometry("1040x620")
        self.minsize(1040, 620)

        self.image_path: Path | None = None
        self.photo_preview = None

        self.configure(bg="#f6f7fb")

        # ===== Header =====
        header = tk.Frame(self, bg="#f6f7fb")
        header.pack(fill="x", padx=16, pady=(14, 8))

        title = tk.Label(
            header,
            text="MedicalNN",
            font=("Arial", 22, "bold"),
            fg="#1f2a44",
            bg="#f6f7fb",
        )
        title.pack(side="left")

        subtitle = tk.Label(
            header,
            text="Локальная система диагностики",
            font=("Arial", 11),
            fg="#55607a",
            bg="#f6f7fb",
        )
        subtitle.pack(side="left", padx=14)

        self.lbl_device = tk.Label(
            header,
            text=get_device_text(),
            font=("Arial", 10),
            fg="#55607a",
            bg="#f6f7fb",
        )
        self.lbl_device.pack(side="right")

        # ===== Main =====
        main = tk.Frame(self, bg="#f6f7fb")
        main.pack(fill="both", expand=True, padx=16, pady=12)

        left_card = tk.Frame(main, bg="white", bd=0, highlightthickness=1, highlightbackground="#e6e8ef")
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_card = tk.Frame(main, bg="white", bd=0, highlightthickness=1, highlightbackground="#e6e8ef")
        right_card.pack(side="right", fill="both", expand=True)

        # ===== Left =====
        left_top = tk.Frame(left_card, bg="white")
        left_top.pack(fill="x", padx=14, pady=12)

        self.btn_open = tk.Button(
            left_top,
            text="Выбрать изображение",
            command=self.select_image,
            font=("Arial", 11),
            padx=10,
            pady=6,
            relief="groove",
        )
        self.btn_open.pack(side="left")

        self.btn_predict = tk.Button(
            left_top,
            text="Диагностировать",
            command=self.run_predict,
            font=("Arial", 11, "bold"),
            padx=12,
            pady=6,
            relief="groove",
            state="disabled",
        )
        self.btn_predict.pack(side="left", padx=10)

        self.lbl_path = tk.Label(
            left_card,
            text="Файл: —",
            font=("Arial", 10),
            fg="#55607a",
            bg="white",
            wraplength=460,
            justify="left",
        )
        self.lbl_path.pack(anchor="w", padx=14, pady=(0, 10))

        self.img_label = tk.Label(
            left_card,
            text="Выберите изображение .jpg/.jpeg/.png",
            font=("Arial", 12),
            fg="#55607a",
            bg="#fafbff",
            bd=0,
            highlightthickness=1,
            highlightbackground="#e6e8ef",
            padx=10,
            pady=10,
        )
        self.img_label.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # ===== Right =====
        right_top = tk.Frame(right_card, bg="white")
        right_top.pack(fill="x", padx=14, pady=12)

        tk.Label(
            right_top,
            text="Результат диагностики",
            font=("Arial", 15, "bold"),
            fg="#1f2a44",
            bg="white",
        ).pack(anchor="w")

        self.lbl_main = tk.Label(
            right_card,
            text="Класс: —    Вероятность: —",
            font=("Arial", 12),
            fg="#1f2a44",
            bg="white",
        )
        self.lbl_main.pack(anchor="w", padx=14, pady=(6, 2))

        self.lbl_note = tk.Label(
            right_card,
            text="",
            font=("Arial", 10),
            fg="#55607a",
            bg="white",
            wraplength=470,
            justify="left",
        )
        self.lbl_note.pack(anchor="w", padx=14, pady=(0, 10))

        tk.Label(
            right_card,
            text="Вероятности принадлежности ко всем классам",
            font=("Arial", 11),
            fg="#55607a",
            bg="white",
        ).pack(anchor="w", padx=14, pady=(2, 8))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", font=("Arial", 11), rowheight=28)
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))

        self.tree = ttk.Treeview(right_card, columns=("cls", "prob"), show="headings", height=12)
        self.tree.heading("cls", text="Класс")
        self.tree.heading("prob", text="Вероятность")
        self.tree.column("cls", anchor="w", width=160)
        self.tree.column("prob", anchor="center", width=130)
        self.tree.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        self.tree.tag_configure("pred", background="#eaf2ff")
        self.tree.tag_configure("mel", background="#ffeaea")

        self.lbl_classes = tk.Label(
            right_card,
            text="",
            font=("Arial", 9),
            fg="#7b85a2",
            bg="white",
            wraplength=470,
            justify="left",
        )
        self.lbl_classes.pack(anchor="w", padx=14, pady=(0, 14))

        # ===== Checks =====
        self._check_model_files()

    def _check_model_files(self):
        missing = []
        if not CFG.best_model_path.exists():
            missing.append(f"Модель: {CFG.best_model_path}")
        if not CFG.label_map_path.exists():
            missing.append(f"label_map: {CFG.label_map_path}")

        if missing:
            self.btn_predict.config(state="disabled")
            messagebox.showwarning(
                "Файлы модели не найдены",
                "Не найдены необходимые файлы:\n\n"
                + "\n".join(missing)
                + "\n\nСначала запусти подготовку данных и обучение.",
            )
            return

        try:
            lm = load_label_map()
            classes = ", ".join(sorted(lm.keys()))
            self.lbl_classes.config(text=f"Классы модели: {classes}")
        except Exception as e:
            self.btn_predict.config(state="disabled")
            messagebox.showerror("Ошибка label_map", f"Не удалось прочитать label_map.json:\n{e}")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Images", "*.jpg *.jpeg *.png")],
        )
        if not file_path:
            return

        self.image_path = Path(file_path)
        self.lbl_path.config(text=f"Файл: {self.image_path}")

        try:
            img = Image.open(self.image_path).convert("RGB")
            img.thumbnail((640, 520))
            self.photo_preview = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.photo_preview, text="")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{e}")
            self.image_path = None
            self.btn_predict.config(state="disabled")
            return

        # Можно предсказывать только если файлы модели есть
        if CFG.best_model_path.exists() and CFG.label_map_path.exists():
            self.btn_predict.config(state="normal")

        self.lbl_main.config(text="Класс: —    Вероятность: —")
        self.lbl_note.config(text="")
        for item in self.tree.get_children():
            self.tree.delete(item)

    def run_predict(self):
        if self.image_path is None:
            return

        # UI feedback
        self.btn_predict.config(state="disabled")
        self.lbl_note.config(text="Выполняется анализ...")

        def worker():
            try:
                result = predict_image(self.image_path)

                # Ожидаемый формат:
                # {"predicted_class": "MEL", "probability": 0.84, "all_probs": {"AK":..., ...}}
                pred = result.get("predicted_class", None)
                prob = float(result.get("probability", 0.0))
                all_probs = result.get("all_probs", None)

                if pred is None or all_probs is None or not isinstance(all_probs, dict):
                    raise ValueError(
                        "predict_image вернул неожиданный формат. "
                        "Ожидается ключи: predicted_class, probability, all_probs(dict)."
                    )

                self.after(0, lambda: self._render_result(pred, prob, all_probs))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка инференса:\n{e}"))
                self.after(0, lambda: self.lbl_note.config(text=""))
            finally:
                self.after(0, lambda: self.btn_predict.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _render_result(self, pred: str, prob: float, all_probs: dict):
        self.lbl_main.config(text=f"Класс: {pred}    Вероятность: {prob:.4f}")

        mel_p = all_probs.get("MEL", None)
        if mel_p is not None:
            mel_p = float(mel_p)
            if mel_p >= 0.70:
                note = "Высокая вероятность MEL (меланомы). Рекомендуется консультация специалиста."
            elif mel_p >= 0.40:
                note = "Умеренная вероятность MEL (меланомы). Рекомендуется дополнительная проверка."
            else:
                note = "Низкая вероятность MEL (меланомы) по оценке модели."
            self.lbl_note.config(text=note)
        else:
            self.lbl_note.config(text="")

        for item in self.tree.get_children():
            self.tree.delete(item)

        sorted_items = sorted(all_probs.items(), key=lambda x: float(x[1]), reverse=True)
        for cls, p in sorted_items:
            p = float(p)
            tags = []
            if cls == pred:
                tags.append("pred")
            if cls == "MEL":
                tags.append("mel")
            self.tree.insert("", "end", values=(cls, f"{p:.4f}"), tags=tuple(tags))


def run():
    app = MedicalNNApp()
    app.mainloop()


if __name__ == "__main__":
    run()