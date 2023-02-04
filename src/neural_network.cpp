#include "neural_network.hpp"

#include <cassert>

neural_network::neural_network(const std::vector<size_t>& topology) {
	layers_.reserve(topology.size());

	for (size_t i = 0; i < topology.size(); ++i) {
		layers_.emplace_back();

		size_t outputs = i == topology.size() - 1 ? 0 : topology[i + 1];

		for (size_t j = 0; j < topology[i]; ++j) {
			layers_.back().emplace_back(neuron(outputs, j));
		}

		layers_.back().back().set_output(1.0);
	}
}

neural_network::~neural_network() {}

void neural_network::feed_forward(const std::vector<double>& input) {
	assert(input.size() == layers_[0].size() - 1);

	for (size_t i = 0; i < input.size(); ++i) {
		layers_[0][i].set_output(input[i]);
	}

	for (size_t i = 1; i < layers_.size(); ++i) {
		auto& prev_layer = layers_[i - 1];

		for (size_t j = 0; j < layers_[i].size() - 1; ++j) {
			layers_[i][j].feed_forward(prev_layer);
		}
	}
}

void neural_network::back_propagation(const std::vector<double>& target) {
	auto& output_layer = layers_.back();

	// Calculate overall net error
	error_ = 0.0;

	for (size_t i = 0; i < output_layer.size() - 1; ++i) {
		auto delta = target[i] - output_layer[i].get_output();

		error_ += delta * delta;
	}

	error_ /= output_layer.size() - 1;
	error_ = sqrt(error_);

	// Implement a recent average measurement
	recent_average_error_ = (recent_average_error_ * recent_average_smoothing_factor_ + error_) /
							(recent_average_smoothing_factor_ + 1.0);

	// Calculate output layer gradients
	for (size_t i = 0; i < output_layer.size() - 1; ++i) {
		output_layer[i].calculate_output_gradients(target[i]);
	}

	// Calculate hidden layer gradients
	for (size_t i = layers_.size() - 2; i > 0; --i) {
		auto& hidden_layer = layers_[i];
		auto& next_layer = layers_[i + 1];

		for (size_t j = 0; j < hidden_layer.size(); ++j) {
			hidden_layer[j].calculate_hidden_gradients(next_layer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
	for (size_t i = layers_.size() - 1; i > 0; --i) {
		auto& layer = layers_[i];
		auto& prev_layer = layers_[i - 1];

		for (size_t j = 0; j < layer.size() - 1; ++j) {
			layer[j].update_input_weights(prev_layer);
		}
	}
}

void neural_network::get_results(std::vector<double>& output) const {
	output.clear();

	output.reserve(layers_.back().size() - 1);

	for (size_t i = 0; i < layers_.back().size() - 1; ++i) {
		output.push_back(layers_.back()[i].get_output());
	}
}

double neural_network::get_recent_average_error() const {
	return recent_average_error_;
}
