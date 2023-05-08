import tkinter as tk 

class Model:
	def __init__(self):
		self.operator = None 
	
	def set_operator(self, operator):
		self.operator = operator 

	def log_system(self):
		

class View:
	def __init__(self, master):
		self.display = tk.Entry(master, width=20)
		self.display.grid(row=0, column=0, columnspan=4, pady=10)

	def 