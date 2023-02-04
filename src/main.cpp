#include "neural_network.hpp"
#include "util.hpp"

#include <renderer/core.hpp>
#include <iostream>

#include <thread>

#include <fmt/format.h>

std::unique_ptr<renderer::win32_window> application;
std::unique_ptr<renderer::d3d11_renderer> dx11;

renderer::sync_manager updated_draw;
renderer::sync_manager updated_buf;

bool update_size = false;
bool close_requested = false;

std::unique_ptr<neural_network> network;

struct training_data {
	std::vector<double> inputs;
	std::vector<double> targets;

	double latest;
};

void draw_thread() {
	std::vector<training_data> data_set = {
		{{0.0, 0.0}, {0.0}, 0.0},
		{{1.0, 0.0}, {1.0}, 0.0},
		{{0.0, 1.0}, {1.0}, 0.0},
		{{1.0, 1.0}, {0.0}, 0.0}
	};

	size_t generation = 0;

	const auto id = dx11->register_buffer();

	auto segoe = dx11->register_font("Segoe UI Emoji", 10, FW_THIN, true);

	while (!close_requested) {
		updated_draw.wait();

		auto buf = dx11->get_working_buffer(id);

		buf->push_font(segoe);

		buf->draw_text({25.0f, 25.0f}, fmt::format("Generation: {}", generation));
		buf->draw_text({25.0f, 45.0f}, fmt::format("Error: {:.12f}", network->get_recent_average_error()));

		// Train network
		for (size_t i = 0; i < 1500; i++) {
			auto& data = data_set[util::random<size_t>(0, 3)];

			network->feed_forward(data.inputs);

			std::vector<double> output;
			network->get_results(output);
			data.latest = output[0];

			network->back_propagation(data.targets);

			generation++;
		}

		// Data data
		for (size_t i = 0; i < data_set.size(); i++) {
			auto& data = data_set[i];

			buf->draw_text({25.0f, 75.0f + i * 75.0f}, fmt::format("Inputs: {}, {}", data.inputs[0], data.inputs[1]));
			buf->draw_text({25.0f, 75.0f + i * 75.0f + 15.0f}, fmt::format("Target: {}", data.targets[0]));
			buf->draw_text({25.0f, 75.0f + i * 75.0f + 30.0f}, fmt::format("Latest: {:.12f}", data.latest));
		}

		// Show network
		for (size_t i = 0; i < network->layers_.size(); i++) {
			auto& layer = network->layers_[i];

			for (size_t j = 0; j < layer.size(); j++) {
				auto& neuron = layer[j];

				glm::vec2 pos = {300.0f + i * 50.0f, 200.0f + j * 50.0f};

				buf->draw_circle_filled(pos, 20.0f, COLOR_WHITE);
				buf->draw_text(pos, fmt::format("{:.2f}", neuron.get_output()), COLOR_BLACK);
			}
		}

		buf->pop_font();

		dx11->swap_buffers(id);
		updated_buf.notify();
	}
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	switch (msg) {
		case WM_CLOSE:
			close_requested = true;
			return 0;
		case WM_SIZE:
			application->set_size({LOWORD(lParam), HIWORD(lParam)});
			update_size = true;
			break;
		default:
			break;
	}

	return DefWindowProcA(hWnd, msg, wParam, lParam);
}

int main() {
	network = std::make_unique<neural_network>(std::vector<size_t>{3, 4, 4, 2});

	application = std::make_unique<renderer::win32_window>();
	application->set_title("D3D11 Renderer");
	application->set_size({720, 600});

	{
		RECT client;
		if (GetClientRect(GetDesktopWindow(), &client)) {
			const auto size = application->get_size();
			application->set_pos({client.right / 2 - size.x / 2, client.bottom / 2 - size.y / 2});
		}
	}

	application->set_proc(WndProc);

	if (!application->create()) {
		MessageBoxA(nullptr, "Failed to create window.", "Error", MB_ICONERROR | MB_OK);
		return 1;
	}

	dx11 = std::make_unique<renderer::d3d11_renderer>(application.get());

	if (!dx11->init()) {
		MessageBoxA(nullptr, "Failed to initialize renderer.", "Error", MB_ICONERROR | MB_OK);
		return 1;
	}

	dx11->set_vsync(false);
	dx11->set_clear_color({88, 88, 88});

	std::thread draw(draw_thread);

	application->set_visibility(true);

	MSG msg{};
	while (!close_requested && msg.message != WM_QUIT) {
		while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		if (msg.message == WM_NULL && !IsWindow(application->get_hwnd())) {
			close_requested = true;
			break;
		}

		if (update_size) {
			dx11->resize();
			dx11->reset();

			update_size = false;
		}

		dx11->draw();

		updated_draw.notify();
		updated_buf.wait();
	}

	draw.join();

	dx11->release();
	application->destroy();

	return 0;
}