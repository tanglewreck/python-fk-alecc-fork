""""Hitta 5 fel i koden"""
import tkinter as tk
import random

class Ball:
    """foo"""
    def __init__(self, canvas, color, radius):
        """foo"""
        self.canvas = canvas
        self.radius = radius
        self.color = color
        self.x_speed = random.randint(1, 5)
        self.y_speed = random.randint(1, 5)
        x = random.randint(self.radius, canvas_width - self.radius)
        y = random.randint(self.radius, canvas_height - self.radius)
        self.ball = self.canvas.create_oval(x - self.radius,
                                            y - self.radius,
                                            x + self.radius,
                                            y + self.radius,
                                            fill=self.color)
        self.canvas.tag_bind(self.ball, '<Button-1>', self.change_color)

    def change_color(self, event):
        """foo"""
        colors = ["red", "green", "blue", "yellow"]
        new_color = random.choice(colors)
        self.canvas.itemconfig(self.ball, new_color)
        self.color = new_color

    def move(self):
        """foo"""
        self.canvas.move(self.ball, self.x_speed, self.y_speed)
        x1, y1, x2, y2 = self.canvas.coords(self.ball)

        if x1 <= 0 or x2 >= canvas_width:
            self.x_speed *= -1
        if y1 <= 0 or y2 >= canvas_height:
            self.y_speed *= -1


def create_new_ball():
    """foo"""
    ball_radius = 20
    ball_color = "#FF0000"
    new_ball = Ball(canvas, ball_color, ball_radius)

def move_ball():
    """foo"""
    ball.move()
    root.after(50, move_ball)

def main():
    """foo"""
    global root, canvas, ball, canvas_width, canvas_height

    # Fönstret
    root = tk.Tk()
    root.title("Bollklickspelet")

    # Canvas
    canvas_width = 500
    canvas_height = 500
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()

    # Skapa boll
    ball_radius = 20
    ball_color = "#yellow"
    ball = Ball(canvas, ball_color, ball_radius)

    # Skapa ny boll
    create_new_ball()

    # Boll 1 rör sig
    move_ball()

    root.mainloop()

if __name__ == "__main__":
    main()
