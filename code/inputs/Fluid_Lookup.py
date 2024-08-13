import random
import CoolProp.CoolProp as CP

from inputs.Inputs import Input, AveragingInput

from inputs.Pressure_Sensors import PressureSensor
from inputs.Temperature_Sensors import TemperatureSensor

def psi_to_pascals(psi: float) -> float:
    '''Convert PSIA to Pascals'''
    return psi*6894.7572932

def F_to_K(f: float) -> float:
    '''Convert Fahrenheit to Kelvin'''
    return (f- 32) * 5/9 + 273.15

def kg_m3_to_lbm_ft3(kg_m3: float) -> float:
    '''Concerts kg/m^3 to lbm/ft^3'''
    return kg_m3 * 0.0624279606

class FluidLookup(Input):
    def __init__(self, sensor1: 'Input', sensor2: 'Input', fluid: str = 'Water', name: str = None):
        super().__init__(true_value=0.0, name=name)
        self.sensor1 = sensor1      # Pressure sensor
        self.sensor2 = sensor2      # Temperature sensor
        self.fluid = fluid

        self.uncertainties = {      # In percent of table value
            0: 0.0001,              # Liquid phase
            1: 0.05,                # Vapor
            2: 0.05,                # Supercritical
            3: 0.05,                # Supercritical gas
            4: 0.05,                # Supercritical liquid
            5: 0.05,                # Two-phase
            6: 0.1,                 # Near critical point
            8: 0.05                 # Solid
        }

    def get_input_errors(self) -> str:
        string = ""
        if isinstance(self.sensor1, Input):
            string += f"{self.sensor1.get_input_errors()}\n"
        if isinstance(self.sensor2, Input):
            string += f"{self.sensor2.get_input_errors()}\n"
        return string
    
    def get_arithmetic(self) -> str:
        lookup_type = type(self).__name__

        return f"{lookup_type}({self.sensor1.get_arithmetic()}, {self.sensor2.get_arithmetic()})"

class DensityLookup(FluidLookup):
    def __init__(self, pressure_sensor: 'PressureSensor', temperature_sensor: 'TemperatureSensor', fluid: str = 'Water', name: str = None):
        if not isinstance(pressure_sensor, PressureSensor):
            if isinstance(pressure_sensor, AveragingInput):
                if not isinstance(pressure_sensor.get_first(), PressureSensor):
                    raise ValueError(f"Sensor 1 for {self.name} must be a PressureSensor or an AveragingInput of a PressureSensor")
            else:
                raise ValueError(f"Sensor 1 for {self.name} must be a PressureSensor or an AveragingInput of a PressureSensor")
        if not isinstance(temperature_sensor, TemperatureSensor):
            if isinstance(temperature_sensor, AveragingInput):
                if not isinstance(temperature_sensor.get_first(), TemperatureSensor):
                    raise ValueError(f"Sensor 1 for {self.name} must be a TemperatureSensor or an AveragingInput of a TemperatureSensor")
            else:
                raise ValueError(f"Sensor 1 for {self.name} must be a TemperatureSensor or an AveragingInput of a TemperatureSensor")
        super().__init__(pressure_sensor, temperature_sensor, fluid, name=name)

    def calc_error(self, pressure: float, temperature: float, density: float) -> float:
        # Check the phase of the fluid to determine the uncertainty according to NIST
        phase_index = int(CP.PropsSI('Phase', 'T', temperature, 'P', pressure, self.fluid))

        uncertainty_percentage = self.uncertainties[phase_index]

        std_dev = density * (uncertainty_percentage / 100)
        
        return random.gauss(0, std_dev)

    def get_reading(self) -> float:
        pressure = psi_to_pascals(self.sensor1.get_reading())               # Convert from psi to Pa
        if pressure < 0:
            pressure = 0.01
        temperature = F_to_K(self.sensor2.get_reading())                 # Convert from F to K
        if temperature < 0:
            temperature = 0.01
        density = CP.PropsSI('D', 'P', pressure, 'T', temperature, self.fluid)
        density = kg_m3_to_lbm_ft3(density)
        
        error = self.calc_error(pressure, temperature, density)

        density += error

        # Append the final value to self.all_readings for final calculations at the end
        self.all_readings.append(density)

        return density
    
    def get_reading_isolating_input(self, input_to_isolate: 'Input'):
        '''Gets true value from input except from the input to isolate'''
        pressure = psi_to_pascals(self.sensor1.get_reading_isolating_input(input_to_isolate))               # Convert from psi to Pa
        if pressure < 0:
            pressure = 0.01
        temperature = F_to_K(self.sensor2.get_reading_isolating_input(input_to_isolate))                 # Convert from F to K
        if temperature < 0:
            temperature = 0.01
        density = CP.PropsSI('D', 'P', pressure, 'T', temperature, self.fluid)
        density = kg_m3_to_lbm_ft3(density)

        # Append the final value to self.all_readings for final calculations at the end
        self.all_readings.append(density)

        return density
    
    def get_true(self) -> float:
        pressure = psi_to_pascals(self.sensor1.get_true())               # Convert from psi to Pa
        if pressure < 0:
            pressure = 0.01
        temperature = F_to_K(self.sensor2.get_true())                 # Convert from F to K
        if temperature < 0:
            temperature = 0.01
        density = CP.PropsSI('D', 'P', pressure, 'T', temperature, self.fluid)
        density = kg_m3_to_lbm_ft3(density)
        
        error = self.calc_error(pressure, temperature, density)

        density += error

        # Append the final value to self.all_readings for final calculations at the end
        self.all_readings.append(density)

        return density
    
    def reset_error_gain(self) -> None:
        self.sensor1.reset_error_gain()
        self.sensor2.reset_error_gain()