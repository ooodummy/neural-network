#include "neuron.hpp"

#include "util.hpp"

#include <cstdlib>

neuron::neuron(size_t outputs, size_t index) : index_(index) {
	output_weights_.reserve(outputs);

	for (size_t i = 0; i < outputs; ++i) {
		output_weights_.emplace_back(connection());
		output_weights_.back().weight = util::random<double>(0.0, 1.0);
	}
}

neuron::~neuron() {}

void neuron::set_output(double output) {
	output_ = output;
}

double neuron::get_output() const {
	return output_;
}

void neuron::feed_forward(const std::vector<neuron>& prev_layer) {
	double sum = 0.0;

	for (size_t i = 0; i < prev_layer.size(); ++i) {
		auto& neuron = prev_layer[i];

		sum += neuron.output_ * neuron.output_weights_[index_].weight;
	}

	output_ = transfer_function(sum);
}

double neuron::transfer_function(double x) {
	return tanh(x);
}

double neuron::transfer_function_derivative(double x) {
	return 1.0 - x * x;
}

void neuron::calculate_output_gradients(double target) {
	auto delta = target - output_;

	gradient_ = delta * transfer_function_derivative(output_);
}

double neuron::sum_dow(const std::vector<neuron>& next_layer) const {
	double sum = 0.0;

	for (size_t i = 0; i < next_layer.size() - 1; ++i) {
		sum += output_weights_[i].weight * next_layer[i].gradient_;
	}

	return sum;
}

void neuron::calculate_hidden_gradients(const std::vector<neuron>& next_layer) {
	double dow = sum_dow(next_layer);

	gradient_ = dow * transfer_function_derivative(output_);
}

void neuron::update_input_weights(std::vector<neuron>& prev_layer) {
	for (size_t i = 0; i < prev_layer.size(); ++i) {
		auto& neuron = prev_layer[i];

		auto old_delta_weight = neuron.output_weights_[index_].delta_weight;

		auto new_delta_weight = eta_ * neuron.output_ * gradient_ + alpha_ * old_delta_weight;

		neuron.output_weights_[index_].delta_weight = new_delta_weight;
		neuron.output_weights_[index_].weight += new_delta_weight;
	}
}