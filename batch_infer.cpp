#include <random> 
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace cv;
using namespace std;

namespace examples {

	void img_classifiction(int max_batch_size) {

		string onnxModelPath = "models/transfer_baseon_mobilenetv2_v4.onnx";
		if (!ccutil::exists(onnxModelPath)) {
			INFOE("onnx model file not exists, run< python plugin_onnx_export.py > to generate onnx model.");
			return;
		}

		INFOW("onnx to trtmodel...");
		string trtModelSavePath = "models/transfer_baseon_mobilenetv2_v4.fp32.engine";
		//TRTBuilder::compileTRT(
		//	TRTBuilder::TRTMode_FP32,
		//	{}, // outputs
		//	16,  // max batch size
		//	TRTBuilder::ModelSource(onnxModelPath),
		//	trtModelSavePath,
		//	{ TRTBuilder::InputDims(max_batch_size, 3, 224, 224) }
		//);
		INFO("done.");

		INFO("load engine...");
		auto engine = TRTInfer::loadEngine(trtModelSavePath);
		if (!engine) {
			INFO("can not load model.");
			return;
		}
		INFO("done.");

		INFO("start benchmark...");
		default_random_engine e(time(0));
		static uniform_int_distribution<unsigned> u(1, max_batch_size);
		vector<Mat> v_img;
		for (int i = 0; i < 10; i++)
		{
			// random batch size
			int batch_size = u(e);

			INFO("loading source images...");
			string img_dir = "imgs";
			vector<cv::String> img_paths;
			cv::glob(img_dir + "\\*.bmp", img_paths);
			for (int i = 0; i < batch_size; i++) {
				std::string img_path = img_paths[i];
				//std::cout << img_path << std::endl;
				v_img.push_back(imread(img_path));
			}
			INFO("done.");

			//INFO("preprocess inputs...");
			auto start = std::chrono::high_resolution_clock::now();	// clock
			float mean[3] = { 0.485, 0.456, 0.406 };
			float std[3] = { 0.229, 0.224, 0.225 };
			engine->input(0)->resize(batch_size);
			for (int i = 0; i < batch_size; ++i) {
				engine->input(0)->setNormMat(i, v_img[i], mean, std);	//put multi images to a Tensor(batching)
			}
			auto m0 = std::chrono::high_resolution_clock::now();	// clock
			engine->input(0).toGPU(true);
			//INFO("done.");

			//INFO("forward...");
			auto m1 = std::chrono::high_resolution_clock::now();	// clock
			engine->forward(true);
			auto m2 = std::chrono::high_resolution_clock::now();	// clock
			//INFO("done.");

			// get result and copy to cpu
			auto output = engine->output(0);
			auto end = std::chrono::high_resolution_clock::now();	// clock
			output->print();
			//std::cout << t << std::endl;

			auto d0 = std::chrono::duration<float, std::milli>(m0 - start).count();
			std::cout << "PreProcess images: " << float(d0) << " ms" << std::endl;
			auto d1 = std::chrono::duration<float, std::milli>(m1 - m0).count();
			std::cout << "Host to Device: " << float(d1) << " ms" << std::endl;
			auto d2 = std::chrono::duration<float, std::milli>(m2 - m1).count();
			std::cout << "Infer: " << float(d2) << " ms" << std::endl;
			auto d3 = std::chrono::duration<float, std::milli>(end - m2).count();
			std::cout << "Device to Host: " << float(d3) << " ms" << std::endl;

			v_img.clear();
		}

		engine->destroy();

		INFO("All Done.");
	}

	void randomLoadImages(string dir, int num_images, vector<Mat> &v_img) {
		vector<cv::String> img_paths;
		cv::glob(dir + "\\*.bmp", img_paths);
		for (int i = 0; i < num_images; i++) {
			std::string img_path = img_paths[i];
			//std::cout << img_path << std::endl;
			v_img.push_back(imread(img_path));
		}

	}

	int randInt(int min, int max) {
		default_random_engine e(time(0));
		static uniform_int_distribution<unsigned> u(min, max);
		return u(e);
	}

	double avg(std::vector<double> &v, bool drop_first = false)
	{
		int start = 0;
		int n = v.size();
		if (drop_first) {
			start = 1;
			n = n - 1;
		}
		int sum = 0;
		for (int i = 0; i < n; i++) {
			sum += v[i];
		}
		return sum / n;
	}

	double dur(std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end) {
		//float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		return double(duration);
	}
};