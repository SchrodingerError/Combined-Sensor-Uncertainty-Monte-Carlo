from inputs.Inputs import Input

class MathFunction(Input):
    pass

class Integration(MathFunction):
    def __init__(self, input: 'Input', time_delta: float, frequency: float, name: str = None):
        super().__init__(true_value=0.0, input_errors={}, name=name)
        self.input: 'Input' = input
        self.time_delta = time_delta
        self.time_step = 1 / frequency

    def get_reading(self) -> float:
        t = 0.0
        sum = 0.0
        while t < self.time_delta:
            sum += self.input.get_reading() * self.time_step
            t += self.time_step
        return sum
    
    def get_true(self) -> float:
        t = 0.0
        sum = 0.0
        while t < self.time_delta:
            sum += self.input.get_true() * self.time_step
            t += self.time_step
        return sum