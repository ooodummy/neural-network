#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>

struct connection {
	double weight;
	double delta_weight;
};

class neuron {
public:
	neuron(size_t outputs, size_t index);
	~neuron();

	void feed_forward(const std::vector<neuron>& prev_layer);

	void calculate_output_gradients(double target);
	void calculate_hidden_gradients(const std::vector<neuron>& next_layer);
	void update_input_weights(std::vector<neuron>& prev_layer);

	void set_output(double output);
	double get_output() const;

private:
	static double transfer_function(double x);
	static double transfer_function_derivative(double x);

	double sum_dow(const std::vector<neuron>& next_layer) const;

	size_t index_;

	double output_;
	std::vector<connection> output_weights_;

	double gradient_;

	constexpr static double eta_ = 0.15;
	constexpr static double alpha_ = 0.2;
};

#endif