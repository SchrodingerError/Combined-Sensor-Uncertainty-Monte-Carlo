from random import gauss

from inputs.Inputs import Input

def F_to_C(f: float) -> float:
    return (f - 32) / 1.8

class TemperatureSensor(Input):
    pass

class RTD(TemperatureSensor):
    def parse_input_errors(self) -> None:
        self.k = self.input_errors.get("k", 2)                      # By default, assume each uncertainty is given as 2 stdev
        self.class_code = self.input_errors.get("class", "A")

        self.max_allowable_class_error = 0

        true_value_celsius = F_to_C(self.true_value)

        # First find the allowable error in Celsius as defined by IEC 60751:2022
        match self.class_code:
            case "AA":
                self.max_allowable_class_error = 0.1 + 0.0017 * abs(true_value_celsius)
            case "A":
                self.max_allowable_class_error = 0.15 + 0.002 * abs(true_value_celsius)
            case "B":
                self.max_allowable_class_error = 0.3 + 0.005 * abs(true_value_celsius)
            case "C":
                self.max_allowable_class_error = 0.6 + 0.01 * abs(true_value_celsius)
        

        # Convert the error from Celsius to Fahrenheit
        self.max_allowable_class_error *= 1.8

        self.error_stdev = self.max_allowable_class_error / self.k

        self.input_errors["error_stdev"] = self.error_stdev
    
    def calc_class_error(self) -> float:
        return gauss(0, self.error_stdev)
    
    def calc_error(self) -> float:
        class_error = self.calc_class_error()
        return (class_error) * self.error_gain

class Thermocouple(TemperatureSensor):
    pass