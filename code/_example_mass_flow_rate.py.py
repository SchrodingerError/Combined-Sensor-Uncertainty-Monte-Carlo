import matplotlib.pyplot as plt
import math

from inputs.Inputs import AveragingInput, PhysicalInput
from inputs.Pressure_Sensors import PressureTransmitter
from inputs.Temperature_Sensors import RTD, Thermocouple
from inputs.Force_Sensors import LoadCell
from inputs.Flow_Speed_Sensors import TurbineFlowMeter
from inputs.Fluid_Lookup import DensityLookup
from inputs.Math_Functions import Integration

from System_Uncertainty_Monte_Carlo import SystemUncertaintyMonteCarlo

from typing import List, Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inputs.Inputs import Input, CombinedInput

        
def plot_sensitivity_analysis(gain_values: list, stdev_values: list, normal_stdev: float, sensor_name: str):
    # Plotting the standard deviation values against the gain settings
        plt.figure(figsize=(10, 6))
        plt.plot(gain_values, stdev_values, marker='o', linestyle='-', color='b')
        plt.axhline(y=normal_stdev, color='r', linestyle='--', label='Normal System STD Dev')
        plt.title(f'Sensitivity Analysis for Error Gain on {sensor_name}')
        plt.xlabel('Error Gain')
        plt.ylabel('Standard Deviation of Output')
        plt.grid(True)
        plt.show()


def main(num_of_tests: int = 1000, num_of_processes: int = 1):
    # Define true values and sensor error ranges
    pt_1_errors = {
        "k": 2,                             # How many standard deviations each of these errors represents (if applicable)
        "fso": 100,                        # full scale output so that stdev can be calculated for errors based on FSO
        "static_error_band": 0.15,          # +-% of FSO
        "repeatability": 0.02,              # +-% of FSO
        "hysteresis": 0.07,                  # +-% of FSO
        "non-linearity": 0.15,          # +-% of FSO
        "temperature_zero_error": 0.005,    # +% of FSO per degree F off from calibration temp
        "temperature_offset": 30,           # Degrees F off from calibration temp  
    }
    
    RTD_1_errors = {
        "k": 3,                         # How many standard deviations each of these errors represents (if applicable)
        "class": "A",                   # class AA, A, B, C
    }

    flow_1_errors = {
        "k": 2,                             # How many standard deviations each of these errors represents (if applicable)
        "fso": 20,                        # full scale output so that stdev can be calculated for errors based on FSO
        "static_error_band": 0.25,          # +-% of FSO
    }

    pipe_diameter_error = {
        "tolerance": 0.05,
        "cte": 8.8E-6,                      # degrees F^-1
        "temperature_offset": -100           # Degrees off from when the measurement was taken
    }

    # Create sensor instances
    pressure_transmitter_1 = PressureTransmitter(25, pt_1_errors, name="PT 1")
    pressure_transmitter_2 = PressureTransmitter(20, pt_1_errors, name="PT 2")
    temperature_sensor_1 = RTD(250, RTD_1_errors, name="RTD 1")

    average_pressure = AveragingInput([pressure_transmitter_1, pressure_transmitter_2])

    # Create DensityLookup instance
    density_lookup = DensityLookup(pressure_sensor=average_pressure, temperature_sensor=temperature_sensor_1)
    flow_speed = TurbineFlowMeter(10, flow_1_errors, "Flow Sensor 1")
    pipe_diameter = PhysicalInput(1, pipe_diameter_error, name="Pipe Diameter")

    # Define the combined sensor
    combined_input: CombinedInput = math.pi * density_lookup * flow_speed *  (pipe_diameter / 12 / 2)**2
    
    monte_carlo = SystemUncertaintyMonteCarlo(combined_input)
    
    monte_carlo.perform_system_analysis(monte_carlo_settings={
        "num_runs": num_of_tests,
        "num_processes": num_of_processes
    })

    sensitivity_analysis_num_runs = num_of_tests // 100

    monte_carlo.perform_sensitivity_analysis(input_to_analyze=flow_speed,
        gain_settings=[1, 10, 10],
        monte_carlo_settings={
        "num_runs": sensitivity_analysis_num_runs,
        "num_processes": num_of_processes
    })
    monte_carlo.perform_sensitivity_analysis(input_to_analyze=pressure_transmitter_1,
        gain_settings=[0.1, 10, 10],
        monte_carlo_settings={
        "num_runs": sensitivity_analysis_num_runs,
        "num_processes": num_of_processes
    })
    monte_carlo.perform_sensitivity_analysis(input_to_analyze=temperature_sensor_1,
        gain_settings=[0.01, 1, 10],
        monte_carlo_settings={
        "num_runs": sensitivity_analysis_num_runs,
        "num_processes": num_of_processes
    })

    monte_carlo.perform_range_analysis(input_to_analyze=pressure_transmitter_1,
        range_settings=[28, 31, 20],
        monte_carlo_settings={
            "num_runs": sensitivity_analysis_num_runs,
            "num_processes": num_of_processes
        }    
    )

    monte_carlo.perform_range_analysis(input_to_analyze=temperature_sensor_1,
        range_settings=[237, 244, 20],
        monte_carlo_settings={
            "num_runs": sensitivity_analysis_num_runs,
            "num_processes": num_of_processes
        }    
    )

    monte_carlo.save_report()
    



if __name__ == "__main__":
    num_of_tests = 1_000_000
    num_of_processes = 10
    main(num_of_tests, num_of_processes)