import threading

import customtkinter as ctk
from transformers import pipeline


MODEL_NAME = "openai-community/gpt2"
TASK = "text-generation"
MAX_NEW_TOKENS = 10
NUM_RETURN_SEQUENCES = 3


class TextGenerationApp(ctk.CTk):
	def __init__(self):
		super().__init__()
		self.title("Text Generation Studio")
		self.geometry("900x700")
		self.minsize(760, 600)

		ctk.set_appearance_mode("System")
		ctk.set_default_color_theme("blue")

		self.text_generator = None

		self._build_ui()

	def _build_ui(self):
		self.grid_columnconfigure(0, weight=1)
		self.grid_rowconfigure(2, weight=1)

		header_frame = ctk.CTkFrame(self, corner_radius=18)
		header_frame.grid(row=0, column=0, padx=24, pady=(24, 12), sticky="ew")
		header_frame.grid_columnconfigure(0, weight=1)

		title_label = ctk.CTkLabel(
			header_frame,
			text="Text Generation Application",
			font=ctk.CTkFont(size=26, weight="bold"),
		)
		title_label.grid(row=0, column=0, padx=20, pady=(20, 4), sticky="w")

		subtitle_label = ctk.CTkLabel(
			header_frame,
			text=f"Generate three continuations with {MODEL_NAME} using the Transformers pipeline.",
			font=ctk.CTkFont(size=14),
		)
		subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

		input_frame = ctk.CTkFrame(self, corner_radius=18)
		input_frame.grid(row=1, column=0, padx=24, pady=12, sticky="ew")
		input_frame.grid_columnconfigure(0, weight=1)

		prompt_label = ctk.CTkLabel(input_frame, text="Prompt", font=ctk.CTkFont(size=15, weight="bold"))
		prompt_label.grid(row=0, column=0, padx=20, pady=(18, 8), sticky="w")

		self.prompt_entry = ctk.CTkEntry(input_frame, placeholder_text="Enter a starting phrase, for example: What if AI")
		self.prompt_entry.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="ew")
		self.prompt_entry.insert(0, "What if AI")

		button_row = ctk.CTkFrame(input_frame, fg_color="transparent")
		button_row.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
		button_row.grid_columnconfigure(0, weight=1)

		self.generate_button = ctk.CTkButton(button_row, text="Generate Text", command=self.generate_text)
		self.generate_button.grid(row=0, column=0, sticky="w")

		self.status_label = ctk.CTkLabel(button_row, text="Ready", font=ctk.CTkFont(size=13))
		self.status_label.grid(row=0, column=1, padx=(16, 0), sticky="e")

		output_frame = ctk.CTkFrame(self, corner_radius=18)
		output_frame.grid(row=2, column=0, padx=24, pady=(12, 24), sticky="nsew")
		output_frame.grid_columnconfigure(0, weight=1)
		output_frame.grid_rowconfigure(1, weight=1)

		output_label = ctk.CTkLabel(output_frame, text="Generated Results", font=ctk.CTkFont(size=15, weight="bold"))
		output_label.grid(row=0, column=0, padx=20, pady=(18, 8), sticky="w")

		self.output_box = ctk.CTkTextbox(output_frame, wrap="word")
		self.output_box.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
		self.output_box.insert("1.0", "Your generated text will appear here.")
		self.output_box.configure(state="disabled")

	def generate_text(self):
		prompt = self.prompt_entry.get().strip()
		if not prompt:
			self._set_status("Enter a prompt first.")
			return

		self.generate_button.configure(state="disabled")
		self._set_status("Generating...")
		threading.Thread(target=self._generate_text_worker, args=(prompt,), daemon=True).start()

	def _generate_text_worker(self, prompt):
		try:
			if self.text_generator is None:
				self.after(0, self._set_status, f"Loading {MODEL_NAME}...")
				self.text_generator = pipeline(task=TASK, model=MODEL_NAME)

			results = self.text_generator(
				prompt,
				max_new_tokens=MAX_NEW_TOKENS,
				num_return_sequences=NUM_RETURN_SEQUENCES,
			)
			formatted_results = []
			for index, result in enumerate(results, start=1):
				formatted_results.append(f"{index}. {result['generated_text']}")
			output_text = "\n\n".join(formatted_results)
			self.after(0, self._update_output, output_text)
		except Exception as error:
			self.after(0, self._update_output, f"Generation failed: {error}")

	def _update_output(self, text):
		self.output_box.configure(state="normal")
		self.output_box.delete("1.0", "end")
		self.output_box.insert("1.0", text)
		self.output_box.configure(state="disabled")
		self.generate_button.configure(state="normal")
		self._set_status("Ready")

	def _set_status(self, message):
		self.status_label.configure(text=message)


if __name__ == "__main__":
	app = TextGenerationApp()
	app.mainloop()