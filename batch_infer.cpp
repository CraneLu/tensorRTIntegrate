
#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace cv;
using namespace std;

namespace examples {

	void onnx() {

		string onnxModelPath = "models/demo.onnx";
		if (!ccutil::exists(onnxModelPath)) {
			INFOE(onnxModelPath + " not exists, run< python plugin_onnx_export.py > to generate onnx model.");
			return;
		}

		INFOW("onnx to trtmodel...");
		string trtModelSavePath = "models/demo.fp32.trtmodel";
		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32,
			{}, // outputs
			4,  // batch size
			TRTBuilder::ModelSource(onnxModelPath),
			trtModelSavePath,
			{ TRTBuilder::InputDims(3, 224, 224) }
		);
		INFO("done.");

		INFO("load model: " + trtModelSavePath);
		auto engine = TRTInfer::loadEngine(trtModelSavePath);
		if (!engine) {
			INFO("can not load model.");
			return;
		}
		INFO("done.");

		INFO("loading source images...");
		vector<Mat> v_img;
		randomLoadImages(v_img);
		INFO("done.");


		INFO("forward...");
		float mean[3] = { 0.485, 0.456, 0.406 };
		float std[3] = { 0.229, 0.224, 0.225 };
		auto input = engine->input();

		// multi batch sample
		int batchSize = v_img.size();
		input->resize(batchSize);
		for (int i = 0; i < batchSize; ++i) {
			input->setNormMatGPU(i, image, mean, std);
		}

		engine->forward();

		// get result and copy to cpu
		engine->output(0)->cpu<float>();
		engine->tensor("hm")->cpu<float>();

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
};