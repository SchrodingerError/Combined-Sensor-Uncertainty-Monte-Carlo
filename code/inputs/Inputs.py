from random import gauss

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inputs.Pressure_Sensors import PressureSensor
    from inputs.Temperature_Sensors import TemperatureSensor
    from inputs.Force_Sensors import ForceSensor



class Input():
    def __init__(self, true_value: float, input_errors: dict = dict(), name: str = None):
        self.true_value = true_value
        self.input_errors = input_errors
        self.name = name if name is not None else type(self).__name__

        self.parse_input_errors()
        self.error_gain: float = 1.0

        # Values to be used for data analysis at the end of the Monte Carlo
        self.all_readings: list[float] = []
        self.overall_average: float = 0.0
        self.overall_uncertainty: float = 0.0


    def __add__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'add')

    def __sub__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'sub')

    def __mul__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'mul')

    def __truediv__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'div')
    
    def __radd__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'add')

    def __rsub__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'sub')

    def __rmul__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'mul')

    def __rtruediv__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'div')
    
    def __pow__(self, exponent) -> 'CombinedInput':
        return CombinedInput(self, exponent, 'pow')
    
    def __rpow__(self, other)-> 'CombinedInput':
        return CombinedInput(other, self, 'pow')

    def get_name(self) -> str:
        return self.name

    def get_input_errors(self) -> str:
        return self.name + "\n" + str(self.input_errors)

    def get_arithmetic(self) -> str:
        return "{" + self.get_name() + "}"

    def set_error_gain(self, new_gain: float) -> None:
        self.error_gain = new_gain

    def parse_input_errors(self) -> None:
        self.k = self.input_errors.get("k", 2)                      # By default, assume each uncertainty is given as 2 stdev
        self.input_errors["k"] = self.k

        # Static error band
        self.fso = self.input_errors.get("fso", 0.0)
        self.input_errors["fso"] = self.fso
        self.static_error_stdev = self.fso * self.input_errors.get("static_error_band", 0.0) / 100 / self.k
        self.input_errors["static_error_stdev"] = self.static_error_stdev

        # Repeatability error
        self.repeatability_error_stdev = self.fso * self.input_errors.get("repeatability", 0.0) / 100 / self.k
        self.input_errors["repeatability_error_stdev"] = self.repeatability_error_stdev

        # Hysteresis error
        self.hysteresis_error_stdev = self.fso * self.input_errors.get("hysteresis", 0.0) / 100 / self.k
        self.input_errors["hysteresis_error_stdev"] = self.hysteresis_error_stdev

        # Non-linearity
        self.nonlinearity_error_stdev = self.fso * self.input_errors.get("non-linearity", 0.0) / 100 / self.k
        self.input_errors["non-linearity_error_stdev"] = self.nonlinearity_error_stdev

        # Thermal error
        self.temperature_offset = self.input_errors.get("temperature_offset", 0.0)
        self.input_errors["temperature_offset"] = self.temperature_offset
        self.thermal_error_offset = self.fso * self.temperature_offset * self.input_errors.get("temperature_zero_error", 0.0) / 100
        self.input_errors["thermal_error_offset"] = self.thermal_error_offset

        
    
    def calc_static_error(self) -> float:
        return gauss(0, self.static_error_stdev)
    def calc_repeatability_error(self) -> float:
        return gauss(0, self.repeatability_error_stdev)
    def calc_hysteresis_error(self) -> float:
        return gauss(0, self.hysteresis_error_stdev)
    def calc_nonlinearity_error(self) -> float:
        return gauss(0, self.nonlinearity_error_stdev)
    def calc_thermal_zero_error(self) -> float:
        return self.thermal_error_offset

    def calc_error(self) -> float:
        static_error = self.calc_static_error()
        repeatability_error = self.calc_repeatability_error()
        hysteresis_error = self.calc_hysteresis_error()
        nonlinearity_error = self.calc_nonlinearity_error()
        thermal_zero_error = self.calc_thermal_zero_error()
        return (static_error + repeatability_error + hysteresis_error + nonlinearity_error + thermal_zero_error) * self.error_gain

    def get_reading(self) -> float:
        '''Apply the pre-specified error and return a realistic value to mimic the reading of a sensor with error.'''
        error: float = self.calc_error()
        noisy_value: float = self.true_value + error

        # Append the final value to self.all_readings for final calculations at the end
        self.all_readings.append(noisy_value)

        return noisy_value

    def get_reading_isolating_input(self, input_to_isolate: 'Input'):
        '''Gets true value from input except from the input to isolate'''
        if self == input_to_isolate:
            return self.get_reading()
        else:
            return self.get_true()
    
    def get_true(self) -> float:
        return self.true_value

    def set_true(self, new_true: float) -> None:
        self.true_value = new_true
        self.parse_input_errors()

    def reset_error_gain(self) -> None:
        self.set_error_gain(1.0)
    
class CombinedInput(Input):
    def __init__(self, sensor1, sensor2, operation):
        self.sensor1: 'Input' = sensor1
        self.sensor2: 'Input' = sensor2
        self.operation = operation

    def get_input_errors(self) -> str:
        string = ""
        if isinstance(self.sensor1, Input):
            string += f"{self.sensor1.get_input_errors()}\n"
        if isinstance(self.sensor2, Input):
            string += f"{self.sensor2.get_input_errors()}\n"
        return string
    
    def get_arithmetic(self) -> str:
        operation_char = ""
        match self.operation:
            case "add":
                operation_char = "+"
            case "sub":
                operation_char = "-"
            case "mul":
                operation_char = "*"
            case "div":
                operation_char = "/"
            case "pow":
                operation_char = "**"
        
        string = f"({self.sensor1.get_arithmetic() if isinstance(self.sensor1, Input) else str(self.sensor1)} "
        string += operation_char
        string += f" {self.sensor2.get_arithmetic() if isinstance(self.sensor2, Input) else str(self.sensor2)})"
        return string
    
    def get_reading(self) -> float:
        reading1 = self.sensor1.get_reading() if isinstance(self.sensor1, Input) or isinstance(self.sensor1, CombinedInput) else self.sensor1
        reading2 = self.sensor2.get_reading() if isinstance(self.sensor2, Input) or isinstance(self.sensor2, CombinedInput) else self.sensor2

        # No need to add error as the error already comes from the sensor.get_reading()

        if self.operation == 'add':
            return reading1 + reading2
        elif self.operation == 'sub':
            return reading1 - reading2
        elif self.operation == 'mul':
            return reading1 * reading2
        elif self.operation == 'div':
            if reading2 == 0:
                raise ZeroDivisionError("Division by zero is undefined.")
            return reading1 / reading2
        elif self.operation == 'pow':
            return reading1 ** reading2
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")
    
    def get_reading_isolating_input(self, input_to_isolate: 'Input'):
        '''Gets true value from every input except the input to isolate'''
        reading1 = self.sensor1.get_reading_isolating_input(input_to_isolate) if isinstance(self.sensor1, Input) or isinstance(self.sensor1, CombinedInput) else self.sensor1
        reading2 = self.sensor2.get_reading_isolating_input(input_to_isolate) if isinstance(self.sensor2, Input) or isinstance(self.sensor2, CombinedInput) else self.sensor2

        if self.operation == 'add':
            return reading1 + reading2
        elif self.operation == 'sub':
            return reading1 - reading2
        elif self.operation == 'mul':
            return reading1 * reading2
        elif self.operation == 'div':
            if reading2 == 0:
                raise ZeroDivisionError("Division by zero is undefined.")
            return reading1 / reading2
        elif self.operation == 'pow':
            return reading1 ** reading2
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")
        
    def get_true(self) -> float:
        reading1 = self.sensor1.get_true() if isinstance(self.sensor1, Input) or isinstance(self.sensor1, CombinedInput) else self.sensor1
        reading2 = self.sensor2.get_true() if isinstance(self.sensor2, Input) or isinstance(self.sensor2, CombinedInput) else self.sensor2

        # No need to add error as the error already comes from the sensor.get_reading()

        if self.operation == 'add':
            return reading1 + reading2
        elif self.operation == 'sub':
            return reading1 - reading2
        elif self.operation == 'mul':
            return reading1 * reading2
        elif self.operation == 'div':
            if reading2 == 0:
                raise ZeroDivisionError("Division by zero is undefined.")
            return reading1 / reading2
        elif self.operation == 'pow':
            return reading1 ** reading2
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")

    def reset_error_gain(self) -> None:
        '''Resets the error gain of all connected inputs to 1.0'''
        if isinstance(self.sensor1, Input):
            self.sensor1.reset_error_gain()
        if isinstance(self.sensor2, Input):
            self.sensor2.reset_error_gain()

    def __add__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'add')

    def __sub__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'sub')

    def __mul__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'mul')

    def __truediv__(self, other) -> 'CombinedInput':
        return CombinedInput(self, other, 'div')
    
    def __pow__(self, other)-> 'CombinedInput':
        return CombinedInput(self, other, 'pow')

    def __radd__(self, other)-> 'CombinedInput':
        return CombinedInput(other, self, 'add')

    def __rsub__(self, other)-> 'CombinedInput':
        return CombinedInput(other, self, 'sub')

    def __rmul__(self, other)-> 'CombinedInput':
        return CombinedInput(other, self, 'mul')

    def __rtruediv__(self, other)-> 'CombinedInput':
        return CombinedInput(other, self, 'div')

    def __rpow__(self, other)-> 'CombinedInput':
        return CombinedInput(other, self, 'pow')

class AveragingInput(Input):
    def __init__(self, sensors: list['Input']):
        if not sensors:
            raise ValueError("The list of sensors cannot be empty")
        
        # Ensure all sensors are of the same parent type
        parent_class = type(sensors[0])
        while not issubclass(parent_class, Input):
            parent_class = parent_class.__bases__[0]
        
        super().__init__(true_value=0.0, input_errors={})
        self.sensors = sensors

    def get_first(self) -> 'Input':
        return self.sensors[0]

    def get_input_errors(self) -> str:
        string = ""
        for sensor in self.sensors:
            if isinstance(sensor, Input):
                string += f"{sensor.get_input_errors()}\n\n"
        return string
    
    def get_arithmetic(self) -> str:
        string = f"Average({self.sensors[0].get_arithmetic()}"
        if len(self.sensors) != 1:
            for sensor in self.sensors[1:]:
                string += ", "
                string += sensor.get_arithmetic()
        return string + ")"
    
    def get_reading(self) -> float:
        total = sum(sensor.get_reading() for sensor in self.sensors)
        average = total / len(self.sensors)

        # No need to add error as the error already comes from the sensor.get_reading()
        
        # Append the final value to self.all_readings for final calculations at the end
        self.all_readings.append(average)

        return average
    
    def get_reading_isolating_input(self, input_to_isolate: 'Input'):
        '''Gets true value from input except from the input to isolate'''
        total = sum(sensor.get_reading_isolating_input(input_to_isolate) for sensor in self.sensors)
        average = total / len(self.sensors)

        # No need to add error as the error already comes from the sensor.get_reading()
        
        # Append the final value to self.all_readings for final calculations at the end
        self.all_readings.append(average)

        return average
    
    def get_true(self) -> float:
        total = sum(sensor.get_true() for sensor in self.sensors)
        average = total / len(self.sensors)
        
        # Append the final value to self.all_readings for final calculations at the end
        self.all_readings.append(average)

        return average

    def reset_error_gain(self) -> None:
        sensor: Input = None
        for sensor in self.sensors:
            sensor.reset_error_gain()

class PhysicalInput(Input):
    '''A numerical input with an associated tolerance (or uncertainty in %). An example is the diameter of an orifice.
    Typically, tolerance represents 3 standard deviations. So, an orifice of 5mm with tolerance of 1mm
    would fall within [4, 6] 99.73% of the time.'''
    def __init__(self, true_value: float, input_errors: dict = dict(), name: str = None):
        super().__init__(true_value, input_errors=input_errors, name=name)
        

    def parse_input_errors(self) -> None:
        self.k = self.input_errors.get("k", 3)                      # By default, assume each uncertainty is given as 3 stdev
        self.input_errors["k"] = self.k
        self.tolerance:float = self.input_errors.get("tolerance", None)
        self.input_errors["tolerance"] = self.tolerance
        self.uncertainty:float = self.input_errors.get("uncertainty", None)
        self.input_errors["uncertainty"] = self.uncertainty
        
        if self.tolerance is not None:
            self.std_dev: float = self.tolerance / self.k             # Because tolerance represents 3 stdev
        else:
            self.std_dev = self.true_value * self.uncertainty / 100
        self.input_errors["std_dev"] = self.std_dev

        # Handle any temperature contract or expansion
        # Set the self.true_value to a new value after contract
        # Set self.measured_value is what we will apply an error to
        self.measured_value = self.true_value

        if "temperature_offset" in self.input_errors:
            self.cte = self.input_errors.get("cte", 8.8E-6)
            self.temperature_offset = self.input_errors.get("temperature_offset", 0)
            expansion_amount = self.measured_value * self.cte * self.temperature_offset
            self.true_value = self.measured_value + expansion_amount

    def get_arithmetic(self) -> str:
        return "{" + str(self.name) + "}"

    def calc_error(self) -> float:
        return gauss(0, self.std_dev)
    
    def get_reading(self) -> float:
        '''Apply the pre-specified error and return a realistic value to mimic the reading of a sensor with error.'''
        error: float = self.calc_error()
        noisy_value: float = self.measured_value + error

        # Append the final value to self.all_readings for final calculations at the end
        self.all_readings.append(noisy_value)

        return noisy_value
        