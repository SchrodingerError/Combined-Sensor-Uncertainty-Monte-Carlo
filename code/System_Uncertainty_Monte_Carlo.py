import multiprocessing
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PDF_Generator import PDFGenerator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inputs.Inputs import Input

class SystemUncertaintyMonteCarlo():
    def __init__(self, system_input: 'Input'):
        self.system_input = system_input
        self.true_value = self.system_input.get_true()

        output_dir = "Output Files"

        # Ensure the output directory is fresh each time
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # Remove the existing directory with all its content

        # Recreate the directory
        os.makedirs(output_dir)

        # Setup the PDF Generator
        self.pdf = PDFGenerator()
        self.pdf.add_page()
        self.pdf_report_file = 'Output Files\\System Uncertainty Report.pdf'

        self._generate_system_setup()
    
    def save_report(self) -> None:
        '''Saves the pdf'''
        self.pdf.make_pdf_from_memory(self.pdf_report_file)
        print(f"Saved PDF to {self.pdf_report_file}")

    def _generate_definitions(self) -> None:
        self.pdf.add_header_memory("Definitions")
        self.pdf.add_text_memory("True System Output Value", bold=True, newline=False)
        self.pdf.add_text_memory("True output of the system without any noise.", indent=1)

        self.pdf.add_text_memory("Average of Monte-Carlo", bold=True, newline=False)
        self.pdf.add_text_memory("Average value of all runs with applied error.", indent=1)

        self.pdf.add_text_memory("Max Error 95% of the Time/max_2std_error", bold=True, newline=False)
        self.pdf.add_text_memory("95% of all readings are within this error of the true value.", indent=1)

        self.pdf.add_text_memory("Max Error 99.73% of the Time/max_3std_error", bold=True, newline=False)
        self.pdf.add_text_memory("99.73% of all readings are within this error of the true value.", indent=1)

        self.pdf.add_text_memory("Isolated Sensitivity Analysis", bold=True, newline=False)
        self.pdf.add_text_memory("Only the specified input has applied error which has an 'error gain' multiplied to only its error. Every other input has zero applied error.", indent=1)

        self.pdf.add_text_memory("Non-Isolated Sensitivity Analysis", bold=True, newline=False)
        self.pdf.add_text_memory("Every input has applied error. The specified input has an 'error gain' multiplied to only its error.", indent=1)

        self.pdf.add_text_memory("Sensitivity Ratio", bold=True, newline=False)
        self.pdf.add_text_memory("The ratio of (specified error gain):(error with error gain = 1).", indent=1)

        self.pdf.add_text_memory("Confidence Level of Regression Fiting", bold=True, newline=False)
        self.pdf.add_text_memory("The regression fits up to a 4th degree polynomial, exponential, or log function. Confidence is 1 / (1st best RMSE):(2nd best RMSE).", indent=1)

        self.pdf.add_text_memory("Range Analysis", bold=True, newline=False)
        self.pdf.add_text_memory("The specified input has its 'true value' swept over a range while other inputs are held constant. Error is applied throughout the system.", indent=1)

    def _generate_system_setup(self) -> None:
        self._generate_definitions()
        
        self.pdf.add_page_memory()
        print("System Governing Equation:")
        self.pdf.add_header_memory("System Governing Equation")

        arithmetic_string = self.system_input.get_arithmetic()
        print(arithmetic_string)

        self.pdf.add_text_memory(arithmetic_string)

        print("\n")
        print("System Error Settings:")
        self.pdf.add_newline_memory()
        self.pdf.add_header_memory("System Error Settings")

        string = "\t"
        string += self.system_input.get_input_errors()
        string = string.replace("\n\n\n", "\n\n")
        string = string.replace("\n", "\n\t")
        string = string.rstrip()  # Remove trailing newlines and other whitespace characters
        print(string)
        print("\n"*3)

        self.pdf.add_text_memory(string)


    def _print_and_save(self, text: str, indent: int = 0) -> None:
        '''Prints a string to the terminal and to a file'''
        indents = "\t" * indent
        print(indents, end="")
        print(text)
        with open(self.report_file, 'a', encoding='utf-8') as f:  # Open file in append mode with UTF-8 encoding
            f.write(indents)
            f.write(text + '\n')
    
    def _calculate_metrics(self, results: list[float]) -> dict:
        results_array = np.array(results)

        # Calculate key statistics
        average = np.mean(results_array)
        average_percent_difference = (average - self.true_value) / self.true_value * 100
        std_dev = np.std(results_array)
        median = np.median(results_array)
        min_val = np.min(results_array)
        max_val = np.max(results_array)
        std_dev_percent = (std_dev / self.true_value) * 100
        mean_absolute_error = np.mean(np.abs(results_array - self.true_value))
        mean_absolute_percent_error = (mean_absolute_error) / self.true_value * 100
        max_2std_error = max(abs(average + 2*std_dev - self.true_value), abs(average - 2*std_dev - self.true_value))
        max_2std_percent_error = max_2std_error / self.true_value * 100
        max_3std_error = max(abs(average + 3*std_dev - self.true_value), abs(average - 3*std_dev - self.true_value))
        max_3std_percent_error = max_3std_error / self.true_value * 100

        # Organize them into a dictionary
        metrics = {
            "true_value": self.true_value,
            "median": median,
            "average": average,
            "average_percent_difference": average_percent_difference,
            "min_val": min_val,
            "max_val": max_val,
            "std_dev": std_dev,
            "std_dev_percent": std_dev_percent,
            "mean_absolute_error": mean_absolute_error,
            "mean_absolute_percent_error": mean_absolute_percent_error,
            "max_2std_error": max_2std_error,
            "max_2std_percent_error": max_2std_percent_error,
            "max_3std_error": max_3std_error,
            "max_3std_percent_error": max_3std_percent_error,
        }


        return metrics



    def _run_system_monte_carlo(self, num_of_tests) -> list[float]:
        results = []
        for _ in range(num_of_tests):
            try:
                value = self.system_input.get_reading()
                results.append(value)
            except:
                continue
        return results
    
    def _run_system_monte_carlo_multiprocessed(self, num_of_tests: int, num_of_processes) -> list[float]:
        if num_of_processes == 1:
            return self._run_system_monte_carlo(num_of_tests)
        
        # Split the number of tests among the processes
        tests_per_process = num_of_tests // num_of_processes
        remaining_tests = num_of_tests % num_of_processes
        
        # Create a list to store the number of tests each process will run
        tasks = [tests_per_process] * num_of_processes
        for i in range(remaining_tests):
            tasks[i] += 1

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_of_processes) as pool:
            # Map the tasks to the worker processes
            results = pool.starmap(self._run_system_monte_carlo, [(task,) for task in tasks])

        # Flatten the list of results
        results = np.array([item for sublist in results for item in sublist])
        results_without_none = np.array([item for item in results if item is not None])

        return results_without_none
    
    def _generate_system_histogram(self, results: list[float], with_outliers: bool = True) -> str:
        results = np.array(results)

        # Calculate IQR and identify outliers
        Q1 = np.percentile(results, 25)
        Q3 = np.percentile(results, 75)
        IQR = Q3 - Q1
        outliers = (results < (Q1 - 1.5 * IQR)) | (results > (Q3 + 1.5 * IQR))

        # Filter results if outliers are not included
        if not with_outliers:
            results = results[~outliers]

        plt.figure(figsize=(18, 6))

        # Histogram of filtered data
        plt.hist(results, bins=100, edgecolor='black', alpha=0.7)
        plt.axvline(x=self.true_value, color='r', linestyle='--', label='True Value')
        plt.title('Histogram of System Results' + (' with Outliers' if with_outliers else ' without Outliers'))
        plt.xlabel('System Output')
        plt.ylabel('Frequency')
        plt.legend()  # Show the legend for the true value line

        # Save the figure to a file
        if not os.path.exists("Output Files\\System"):
            os.makedirs("Output Files\\System")  # Create the directory if it does not exist
        filename = "Output Files\\System\\Histogram" + (' with Outliers' if with_outliers else ' without Outliers') + ".png"
        print(f"\tSaving '{filename}'")
        plt.savefig(filename, dpi=500)
        plt.close()

        return filename

    def _generate_system_scatter_plot(self, results: list[float], with_outliers: bool = True) -> str:
        '''Makes a scatter plot and saves. Returns the file name it saves to'''
        results = np.array(results)

        # Calculate IQR and identify outliers
        Q1 = np.percentile(results, 25)
        Q3 = np.percentile(results, 75)
        IQR = Q3 - Q1
        outliers = (results < (Q1 - 1.5 * IQR)) | (results > (Q3 + 1.5 * IQR))

        # Filter results if outliers are not included
        if not with_outliers:
            results_filtered = results[~outliers]
        
        # Check if the number of results is more than 1000, randomly sample if so
        if len(results) > 1000:
            results_filtered = np.random.choice(results, 1000, replace=False)
        else:
            results_filtered = results

        # Generate indices for x-axis
        indices = np.arange(len(results_filtered))

        # Calculate mean and standard deviations
        mean = np.mean(results)
        std_dev = np.std(results)

        # Scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(indices, results_filtered, alpha=0.6, marker='o')
        plt.axhline(y=self.true_value, color='r', linestyle='--', label='True Value')
        plt.axhline(y=mean + std_dev, color='black', linestyle='--', label='1st Dev')
        plt.axhline(y=mean - std_dev, color='black', linestyle='--')
        plt.title('Scatter Plot of System Results' + (' with Outliers' if with_outliers else ' without Outliers'))
        plt.xlabel('Run #')
        plt.ylabel('System Output')
        plt.legend()
        
        # Save the figure to a file
        if not os.path.exists("Output Files\\System"):
            os.makedirs("Output Files\\System")  # Create the directory if it does not exist
        filename = "Output Files\\System\\Scatter" + (' with Outliers' if with_outliers else ' without Outliers') + ".png"
        print(f"\tSaving '{filename}'")
        plt.savefig(filename, dpi=500)
        plt.close()

        return filename

    def _generate_system_report(self, metrics: dict, indent: int = 0) -> None:
        self.pdf.add_metrics_list_memory(metrics, indent)

    def perform_system_analysis(self, monte_carlo_settings:dict = dict()) -> None:
        num_of_tests: int = monte_carlo_settings.get("num_runs", 1_000)
        num_of_processes: int = monte_carlo_settings.get("num_processes", 10)

        print(f"System Monte-Carlo Results:")

        self.pdf.add_page_memory()
        self.pdf.add_header_memory("Entire System Text Results", level=2)


        results = self._run_system_monte_carlo_multiprocessed(num_of_tests, num_of_processes)

        metrics = self._calculate_metrics(results)
        self._generate_system_report(metrics, indent=1)

        print(f"Plotting Entire Monte-Carlo Results:")

        self.pdf.add_page_memory()
        self.pdf.add_header_memory("Entire System Histograms", level=2)
        filename = self._generate_system_histogram(results, with_outliers=True)
        self.pdf.add_centered_image_memory(filename, width_ratio=1.15)

        filename = self._generate_system_histogram(results, with_outliers=False)
        self.pdf.add_centered_image_memory(filename, width_ratio=1.15)

        self.pdf.add_page_memory()
        self.pdf.add_header_memory("Entire System Scatter Plots", level=2)
        filename = self._generate_system_scatter_plot(results, with_outliers=True)
        self.pdf.add_centered_image_memory(filename, width_ratio=1)

        self._generate_system_scatter_plot(results, with_outliers=False)
        self.pdf.add_centered_image_memory(filename, width_ratio=1)

        print("\n")






    def _fit_to_polynomial(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[str, float, str]:
        """
        Fit the given x and y values to various models up to a polynomial degree, determine which fits best, 
        provide a confidence measure based on RMSE comparisons, and return the best fit function as a string up to the degree of the best model.
        
        Parameters:
            x_values (np.ndarray): Array of x values, must be positive for ln(x) and all y_values must be positive for exponential fit.
            y_values (np.ndarray): Array of y values, must be positive for the exponential and logarithmic fits.
        
        Returns:
            tuple: The best fitting model type, a confidence score, and the best fit function as a string.
        """
        
        models = {
            "constant": np.ones_like(x_values),
            "linear": x_values,
            "quadratic": x_values**2,
            "cubic": x_values**3,
            "quartic": x_values**4,
            "exponential": x_values,  # will use log(y) for fitting
            "logarithmic": np.log(x_values)
        }
        
        best_model = None
        min_rmse = np.inf
        rmse_values = {}
        coefficients = {}
        
        # Initial fit to find the best model type
        for model_name, model_values in models.items():
            # Prepare the response variable and design matrix
            if model_name == "exponential":
                transformed_y = np.log(y_values)
            else:
                transformed_y = y_values
            
            A = np.column_stack((np.ones_like(x_values), model_values))
            
            try:
                # Solve the least squares problem
                coeffs, _, _, _ = np.linalg.lstsq(A, transformed_y, rcond=None)
                coefficients[model_name] = coeffs  # Store coefficients
                
                # Predict y values using the model
                if model_name == "exponential":
                    y_pred = np.exp(A @ coeffs)
                else:
                    y_pred = A @ coeffs

                # Calculate RMSE
                rmse = np.sqrt(np.mean((y_values - y_pred)**2))
                rmse_values[model_name] = rmse
                
                # Update best model if current RMSE is lower
                if rmse < min_rmse:
                    min_rmse = rmse
                    best_model = model_name
            except np.linalg.LinAlgError:
                print(f"SVD did not converge for the {model_name} model.")
                continue

        # Construct a new polynomial up to the degree of the best model, if it's a polynomial
        indexes = ["constant", "linear", "quadratic", "cubic", "quartic"]
        if best_model in ["constant", "linear", "quadratic", "cubic", "quartic"]:
            degree = indexes.index(best_model)
            model_terms = np.column_stack([x_values**i for i in range(degree + 1)])
            coeffs, _, _, _ = np.linalg.lstsq(model_terms, y_values, rcond=None)
            
            # Generate the function string
            function_str = " + ".join(f"{coeffs[i]:.4f}*x^{i}" if i > 0 else f"{coeffs[i]:.4f}" for i in range(degree + 1))
            
        elif best_model == "exponential":
            function_str = f"{coefficients[best_model][0]:.4f} + {coefficients[best_model][1]:.4f}*e^x"
        elif best_model == "logarithmic":
            function_str = f"{coefficients[best_model][0]:.4f} + {coefficients[best_model][1]:.4f}*ln(x)"
        
        # Calculate confidence measure
        rmse_values_list = [value for value in rmse_values.values()]
        rmse_values_list.sort()
        # average_rmse = (np.sum(list(rmse_values.values())) - min_rmse) / (len(rmse_values) - 1)
        ratio = rmse_values_list[1] / min_rmse if min_rmse != 0 else 0
        confidence = ratio * (1 / (ratio + 1))

        return best_model, confidence, function_str

    def _generate_plot(self, x, x_label, y, y_label, title, directory: str, log=False) -> str:
        # Function to create plots
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it does not exist

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, marker='o')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)

        if log: # Set y-axis to logarithmic scale
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
            
        filename = f"{directory}\\{title}{' log' if log else ''}.png"
        plt.savefig(filename, dpi=500)  # Save the figure
        plt.close()  # Close the plot to free up memory
        print(f"\tSaving '{filename}'",)
        return filename

    def _generate_sensitivity_plots(self, gain_values: np.ndarray, metrics_list: list[dict], name: str, directory: str) -> list[float]:
        # Finding the index for gain_value = 1
        index_gain_one = np.where(gain_values == 1)[0][0]

        # Prepare data for plotting
        max_2std_error = [m['max_2std_error'] for m in metrics_list]
        max_2std_percent_error = [m['max_2std_percent_error'] for m in metrics_list]
        
        # Calculate ratios
        max_2std_error_ratio = max_2std_error / max_2std_error[index_gain_one]

        # Function to create plots
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it does not exist

        # Plotting absolute metrics
        filename_1 = self._generate_plot(gain_values, "Error Gain", max_2std_percent_error, 'Max 2σ Error Percent (%)', f"Sensitivity Analysis on {name}", directory, log=False)
        filename_2 = self._generate_plot(gain_values, "Error Gain", max_2std_percent_error, 'Max 2σ Error Percent (%)', f"Sensitivity Analysis on {name}", directory, log=True)

        # Plotting relative metrics (ratios)
        filename_3 = self._generate_plot(gain_values, "Error Gain", max_2std_error_ratio, 'Max 2σ Error Ratio', f"Sensitivity Ratio Analysis on {name}", directory, log=False)
        filename_4 = self._generate_plot(gain_values, "Error Gain", max_2std_error_ratio, 'Max 2σ Error Ratio', f"Sensitivity Ratio Analysis on {name}", directory, log=True)

        return [filename_1, filename_2, filename_3, filename_4]

    def _generate_gain_values(self, gain_settings) -> np.ndarray: 
        min_gain = gain_settings[0]
        max_gain = gain_settings[1]
        num_points = gain_settings[2]

        if min_gain < 1 < max_gain:
            # Calculate the number of points for each segment
            num_points_each = num_points // 2
            
            # Generate uniformly spaced values between min_gain and 1
            lower_half = np.linspace(min_gain, 1, num_points_each, endpoint=False)
            
            # Generate uniformly spaced values between 1 and max_gain
            upper_half = np.linspace(1, max_gain, num_points_each, endpoint=True)
            
            # Combine both halves
            gain_values = np.concatenate((lower_half, upper_half))
        else:
            # Generate uniformly spaced values as initially intended
            gain_values = np.linspace(min_gain, max_gain, num_points)
        
        # Add 1 to the list if not there
        if 1 not in gain_values:
            gain_values = np.sort(np.append(gain_values, 1))

        return gain_values

    def _run_isolated_monte_carlo(self, input_to_isolate: 'Input', num_of_tests: int) -> list[float]:
        results = []
        for _ in range(num_of_tests):
            try:
                value = self.system_input.get_reading_isolating_input(input_to_isolate)
                results.append(value)
            except:
                continue
        return results

    def _run_isolated_monte_carlo_multiprocessed(self, input_to_isolate: 'Input', num_of_tests: int, num_of_processes) -> list[float]:
        if num_of_processes == 1:
            return self._run_isolated_monte_carlo(input_to_isolate, num_of_tests)
        
        # Split the number of tests among the processes
        tests_per_process = num_of_tests // num_of_processes
        remaining_tests = num_of_tests % num_of_processes
        
        # Create a list to store the number of tests each process will run
        tasks = [tests_per_process] * num_of_processes
        for i in range(remaining_tests):
            tasks[i] += 1

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_of_processes) as pool:
            # Map the tasks to the worker processes
            results = pool.starmap(self._run_isolated_monte_carlo, [(input_to_isolate, task) for task in tasks])

        # Flatten the list of results
        results = np.array([item for sublist in results for item in sublist])
        results_without_none = np.array([item for item in results if item is not None])

        return results_without_none

    def perform_sensitivity_analysis(self, input_to_analyze: 'Input', gain_settings: list[float], monte_carlo_settings:dict = dict()) -> None:
        self.system_input.reset_error_gain()

        num_of_tests: int = monte_carlo_settings.get("num_runs", 1_000)
        num_of_processes: int = monte_carlo_settings.get("num_processes", 10)

        input_name = input_to_analyze.get_name()

        # Perform isolated analysis where only this input has error
        print(f"Isolated Sensitivity Analysis on {input_name}:")
        metrics_list: list = []
        gain_values = self._generate_gain_values(gain_settings)

        for i in range(len(gain_values)):
            gain_value = gain_values[i]
            input_to_analyze.set_error_gain(gain_value)
            results = self._run_isolated_monte_carlo_multiprocessed(input_to_analyze, num_of_tests, num_of_processes)
            metrics = self._calculate_metrics(results)
            metrics_list.append(metrics)

            print(f"\tError Gain of {gain_value}")

        self.pdf.add_page_memory()
        self.pdf.add_header_memory(f"Isolated Sensitivity Analysis on {input_name}", level=1)
        self.pdf.add_metrics_table_memory(f"Impact of {input_name} Noise Gain on System without Noise", metrics_list, gain_values)

        directory = f"Output Files\\{input_name}\\Isolated"
        print(f"\nPlotting Isolated Sensitivity for {input_name}:")
        filenames: list = self._generate_sensitivity_plots(gain_values, metrics_list, input_name, directory)

        self.pdf.add_page_memory()
        self.pdf.add_header_memory(f"Isolated Sensitivity Analysis on {input_name} Plots", level=1)
        for i in range(len(filenames)):
            if i % 2 == 0:
                self.pdf.add_centered_image_memory(filenames[i], width_ratio=1)
            else:
                self.pdf.add_newline_memory()

        values = [metric["max_2std_percent_error"] for metric in metrics_list]
        polynomial, confidence, eq_string = self._fit_to_polynomial(gain_values, values)

        regression_string = f"Relationship between sensor Error Gain and Max 2σ Error Percent (%) is: {polynomial} with a confidence level of {confidence*100:.2f} %."
        regression_string += f"\nThe equation of best fit is: {eq_string}."
        self.pdf.add_text_memory(regression_string)
        print()


        # Perform non-isolated analysis where only this input has error
        print(f"Non-Isolated Sensitivity Analysis on {input_name}:")
        metrics_list: list = []
        self.system_input.reset_error_gain()
        for i in range(len(gain_values)):
            gain_value = gain_values[i]
            input_to_analyze.set_error_gain(gain_value)
            results = self._run_system_monte_carlo_multiprocessed(num_of_tests, num_of_processes)
            metrics = self._calculate_metrics(results)
            metrics_list.append(metrics)
            print(f"\tError Gain of {gain_value}")

        self.pdf.add_page_memory()
        self.pdf.add_header_memory(f"Non-Isolated Sensitivity Analysis on {input_name}", level=1)
        self.pdf.add_metrics_table_memory(f"Impact of {input_name} Noise Gain on System with Noise", metrics_list, gain_values)


        directory = f"Output Files\\{input_name}\\Non-Isolated"
        print(f"\nPlotting Non-Isolated Sensitivity for {input_name}:")
        filenames: list = self._generate_sensitivity_plots(gain_values, metrics_list, input_name, directory)

        self.pdf.add_page_memory()
        self.pdf.add_header_memory(f"Non-Isolated Sensitivity Analysis on {input_name} Plots", level=1)
        for i in range(len(filenames)):
            if i % 2 == 0:
                self.pdf.add_centered_image_memory(filenames[i], width_ratio=1)

        
        values = [metric["max_2std_percent_error"] for metric in metrics_list]
        polynomial, confidence, eq_string = self._fit_to_polynomial(gain_values, values)
    
        regression_string = f"Relationship between sensor Error Gain and Max 2σ Error Percent (%) is: {polynomial} with a confidence level of {confidence*100:.2f} %."
        regression_string += f"\nThe equation of best fit is: {eq_string}."
        self.pdf.add_text_memory(regression_string)

        # Print the text report of the metrics
        print("\n"*3)







    def _plot_range_analysis(self, points, metrics_list, true_values, sensor_name, directory: str) -> str:
        averages = [metrics['average'] for metrics in metrics_list]
        std_devs = [metrics['std_dev'] for metrics in metrics_list]
        
        upper_bound_1std = [avg + std for avg, std in zip(averages, std_devs)]
        lower_bound_1std = [avg - std for avg, std in zip(averages, std_devs)]
        

        upper_bound_1std = [avg + std for avg, std in zip(averages, std_devs)]
        lower_bound_1std = [avg - std for avg, std in zip(averages, std_devs)]
    
        plt.figure(figsize=(12, 6))
        plt.plot(points, true_values, label='True Value', color="red", linestyle="--")
        plt.plot(points, averages, label='Average', color='blue')
        plt.plot(points, upper_bound_1std, label='1st Dev', color='black', linestyle='--')
        plt.plot(points, lower_bound_1std, color='black', linestyle='--')

        plt.xlabel(f'{sensor_name} Input Value')
        plt.ylabel('System Output')
        plt.title(f"Range Analysis on {sensor_name}")
        plt.legend()
        plt.grid(True)

        # Save the figure
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it does not exist
        filename = f"{directory}\\Range Analysis on {sensor_name}.png"
        plt.savefig(filename, dpi=500)
        plt.close()

        print(f"Range analysis plot saved to '{filename}'")

        return filename


    def perform_range_analysis(self, input_to_analyze: 'Input', range_settings: list[float], monte_carlo_settings:dict = dict()) -> None:
        self.system_input.reset_error_gain()

        num_of_tests: int = monte_carlo_settings.get("num_runs", 1_000)
        num_of_processes: int = monte_carlo_settings.get("num_processes", 10)

        input_name = input_to_analyze.get_name()
        directory = f"Output Files\\{input_name}\\Range Analysis"

        # Perform isolated analysis where only this input has error
        print(f"Range Analysis on {input_name}:")

        # get the original true value to return back to
        original_true_value = input_to_analyze.get_true()

        # Generate a uniform list using range_settings
        min_val, max_val, num_points = range_settings
        points = np.linspace(min_val, max_val, num_points)

        metrics_list = []
        true_values = []

        for point in points:
            print(f"\tTest Value is  {point}")
            # Set the input to the current point in the range
            input_to_analyze.set_true(point)
            
            true_value = self.system_input.get_true()
            true_values.append(true_value)

            # Run the Monte Carlo simulation for the current point
            results = self._run_system_monte_carlo_multiprocessed(num_of_tests, num_of_processes)

            # Calculate metrics for the current results
            metrics = self._calculate_metrics(results)
            metrics_list.append(metrics)

        # Plot the results
        filename = self._plot_range_analysis(points, metrics_list, true_values, input_name, directory)

        # Reset the original true value
        input_to_analyze.set_true(original_true_value)

        self.pdf.add_page_memory()
        self.pdf.add_header_memory(f"Range Analysis on {input_name}", level=1)
        self.pdf.add_centered_image_memory(filename, width_ratio=1)
        self.pdf.add_newline_memory()

        print()