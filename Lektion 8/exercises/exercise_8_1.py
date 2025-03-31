"""exercise 8_1"""

import random
from tkinter import Canvas
from tkinter import Tk

CANVAS_WIDTH = 500
CANVAS_HEIGHT = 500
COLOURS = ["red", "green", "blue", "yellow"]
NUMBER_OF_BALLS = 10

class Ball:
    """Ball class"""
    def __init__(self, canvas, colour, radius):
        """constructor"""
        self.canvas = canvas
        self.radius = radius
        self.colour = colour
        self.x_speed = random.randint(1, 5)
        self.y_speed = random.randint(1, 5)
        x = random.randint(self.radius, CANVAS_WIDTH - self.radius)
        y = random.randint(self.radius, CANVAS_HEIGHT - self.radius)
        self.ball = self.canvas.create_oval(x - self.radius, y - self.radius,
                                            x + self.radius, y + self.radius, fill=self.colour)
        # Bind mousebutton 1 to the colour changing method
        self.canvas.tag_bind(self.ball, '<Button-1>', self.change_colour)

    def change_colour(self, _ = None):
        """Change ball colour, randomly"""
        new_colour = random.choice(COLOURS)
        self.canvas.itemconfig(self.ball, fill=new_colour)
        self.colour = new_colour

        # Set a timer so the ball keeps changing colour
        self.canvas.after(1000, self.change_colour, None)

    def move(self):
        """Move the ball"""
        self.canvas.move(self.ball, self.x_speed, self.y_speed)
        # Get new coordinates
        x1, y1, x2, y2 = self.canvas.coords(self.ball)

        # If outside the canvas, reverse direction
        if x1 <= 0 or x2 >= CANVAS_WIDTH:
            self.x_speed *= -1
        if y1 <= 0 or y2 >= CANVAS_HEIGHT:
            self.y_speed *= -1

        # Set the ball moving again after a delay
        self.canvas.after(50, self.move)



def main():
    """main()"""

    # root widget
    root = Tk()
    root.title("Bollklickspelet")

    # Create a canvas where the balls can move around
    canvas = Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
    canvas.pack()

    # Initialise list (of balls)
    balls = []

    # Add balls to the list
    ball_radius = 20
    ball_colour = "#1122FF"
    for _ in range(NUMBER_OF_BALLS):
        balls.append(Ball(canvas, ball_colour, ball_radius))


    # Set the balls in motion
    for ball in balls:
        ball.move()

    # Execute the mainloop
    root.mainloop()

if __name__ == "__main__":
    main()
