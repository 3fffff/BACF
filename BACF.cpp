#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <filesystem>

using namespace cv;
using namespace std;

struct  Params {

	//filter settings
	cv::Size model_sz = cv::Size(50, 50);
	float target_padding = 2.0;
	//learning parameters
	float update_rate = 0.003;
	float sigma_factor = 1.0 / 16.0;

	//scale settings
	float scale_step = 1.05;
	int num_scales = 1;

}p;

struct TrackedRegion {
	TrackedRegion() { }
	TrackedRegion(const cv::Point2i init_center, const cv::Size init_size) : center(init_center), size(init_size) { }
	TrackedRegion(const cv::Rect box) : center(box.x + round((float)box.size().width / 2.0),
		box.y + round((float)box.size().height / 2.0)),
		size(box.size()) { }

	cv::Rect Rect() const {
		return cv::Rect(center.x - floor((float)size.width / 2.0),
			center.y - floor((float)size.height / 2.0),
			size.width, size.height);
	}

	TrackedRegion resize(const float factor) const {
		TrackedRegion newRegion;
		newRegion.center = center;
		newRegion.size = cv::Size(round(size.width * factor),
			round(size.height * factor));
		return newRegion;
	}

	cv::Point2i center;
	cv::Size size;
};

class BACF {
public:

	/*
	* Initialize the tracker on the region specified in the image
	*/
	void initialize(const cv::Mat& image, const cv::Rect region);
	/*
	*  Return the current bounding box of the target as estimated by the tracker
	*/
	cv::Rect getBoundingBox() const;

	/*
	*  Update the current estimate of the targets position from the image with the current bounding box estimate
	*/
	void detect(const cv::Mat& image);

	/*
	* Update the current tracker model, from the current best position estimated by the tracker in the image provided
	*/
	void update(const cv::Mat& image);

	//protected:
	bool initImpl(const Mat& image, const Rect2d& boundingBox);
	bool updateImpl(const Mat& image, Rect2d& boundingBox);
private:

	//internal functions
	std::vector<cv::Mat> fft2(const cv::Mat featureData);

	int shift_index(const int index, const int length) const;
	cv::Mat make_labels(const cv::Size matrix_size, const cv::Size target_size, const float sigma_factor) const;

	void compute_ADMM();

	cv::Mat compute_response(const std::vector<cv::Mat>& filter, const std::vector<cv::Mat>& sample);

	std::vector<cv::Mat> compute_feature_vec(const cv::Mat& patch);
	void update_impl(const cv::Mat& image, const TrackedRegion& region, const float update_rate);
	cv::Mat detect_impl(const cv::Mat& image, const TrackedRegion& region);
	cv::Mat channelMultiply(std::vector<cv::Mat> a, std::vector<cv::Mat> b, int flags, bool conjb);
	void multiply(cv::Mat cn1, cv::Mat cn2, cv::Mat& result);
	void divide(cv::Mat cn1, cv::Mat cn2, cv::Mat& result);
	std::pair<int, int> minMaxLoc(cv::Mat array);
	cv::Mat BACF::hann_window(int width, int height) const;
	cv::Mat extractTrackedRegion(const cv::Mat image, const TrackedRegion region, const cv::Size output_sz);
	cv::Mat extractTrackedRegionSpec(cv::Mat model, const cv::Size output_sz);
	//parameters set on construction

	//internal state variables
	cv::Mat labelsf; //label function
	cv::Mat window; //cos (hann) window
	std::vector<cv::Mat> model_xf; //regularization marix
	std::vector<cv::Mat> filterf;

	float scale_factor;
	TrackedRegion target;

}; //end definition


bool BACF::initImpl(const Mat& image, const Rect2d& boundingBox) {
	initialize(image, boundingBox);
	return true;
}

bool BACF::updateImpl(const Mat& image, Rect2d& boundingBox) {
	detect(image);
	cv::Rect new_bounding_box = getBoundingBox();
	update(image);
	boundingBox = new_bounding_box;
	return true;
}

void BACF::initialize(const cv::Mat& image, const cv::Rect region) {
	//convert region into internal representation defined from center pixel and size
	//including the padding
	target = TrackedRegion(region).resize(p.target_padding);

	scale_factor = sqrt((float)target.size.area() / (float)p.model_sz.area());

	float resize_factor = (1.0 / scale_factor) * (1.0 / p.target_padding);
	TrackedRegion resized_target = target.resize(resize_factor);

	//create labels
	labelsf = make_labels(p.model_sz, resized_target.size, p.sigma_factor);

	//create window function
	window = hann_window(p.model_sz.width, p.model_sz.height);

	//create the initial filter from the init patch
	update_impl(image, target, 1.0);
}

void BACF::update(const cv::Mat& image) {
	update_impl(image, target, 2);
}

void BACF::multiply(cv::Mat cn1, cv::Mat cn2, cv::Mat& result)
{
	result = cv::Mat::zeros(cn1.size(), cn1.type());

	for (int x = 0; x < result.rows; x++)
		for (int y = 0; y < result.cols; y++)
			result.at<float>(x, y) = cn1.at<float>(x, y) * cn2.at<float>(x, y);
}

void BACF::divide(cv::Mat cn1, cv::Mat cn2, cv::Mat& result)
{
	result = cv::Mat::zeros(cn1.size(), cn1.type());

	for (int x = 0; x < result.rows; x++)
		for (int y = 0; y < result.cols; y++)
			result.at<float>(x, y) = cn1.at<float>(x, y) / cn2.at<float>(x, y);
}

std::pair<int, int> BACF::minMaxLoc(cv::Mat array)
{
	float max = -999999.0f;
	std::pair<int, int> maxpos;
	for (int x = 0; x < p.model_sz.width; x++)
	{
		for (int y = 0; y < p.model_sz.height; y++)
		{
			float val = array.at<float>(y, x);
			if (max < val)
			{
				max = val;
				maxpos.first = x;
				maxpos.second = y;
			}
		}
	}
	return maxpos;
}

cv::Mat BACF::hann_window(int width,int height) const
{
	cv::Mat hann_window(width, height, CV_32F);
	float* tvecx = new float[width + p.target_padding];
	float* tvecy = new float[height + p.target_padding];
	for (int i = 0; i < round((width + p.target_padding) / 2); i++)
	{
		float d = 2.0f * 3.14592653589793f * (float(i) / float(width + 1));
		float hann = 0.5 - 0.5 * std::cos(d);
		tvecx[i] = hann;
		tvecx[width + 1 - i] = hann;
	}
	for (int i = 0; i < round((height + p.target_padding) / 2); i++)
	{
		float d = 2.0f * 3.14592653589793f * (float(i) / float(height + 1));
		float hann = 0.5 - 0.5 * std::cos(d);
		tvecy[i] = hann;
		tvecy[height + 1 - i] = hann;
	}
	for (int x = 1; x < width + 1; x++)
		for (int y = 1; y < height + 1; y++)
			hann_window.at<float>(y - 1, x - 1) = tvecx[x] * tvecy[y];
	delete[] tvecx;
	delete[] tvecy;
	return hann_window;
}
//private functions
std::vector<cv::Mat> BACF::compute_feature_vec(const cv::Mat& patch) {

	//convert the data type to float
	cv::Mat feature_data;

	patch.convertTo(feature_data, CV_32FC1, 1.0 / 255.0, -0.5);

	std::vector<cv::Mat> feature_vec = fft2(feature_data);

	return feature_vec;
}

cv::Rect BACF::getBoundingBox() const {
	TrackedRegion return_bb = target.resize(1.0 / p.target_padding);
	return return_bb.Rect();
}

cv::Mat BACF::extractTrackedRegion(const cv::Mat image, const TrackedRegion region, const cv::Size output_sz) {

	int xMin = region.center.x - floor(((float)region.size.width) / 2.0);
	int yMin = region.center.y - floor(((float)region.size.height) / 2.0);

	int xMax = xMin + region.size.width;
	int yMax = yMin + region.size.height;

	int xMinPad, xMaxPad, yMinPad, yMaxPad;

	if (xMin < 0) {
		xMinPad = -xMin;
	}
	else {
		xMinPad = 0;
	}

	if (xMax > image.size().width) {
		xMaxPad = xMax - image.size().width;
	}
	else {
		xMaxPad = 0;
	}

	if (yMin < 0) {
		yMinPad = -yMin;
	}
	else {
		yMinPad = 0;
	}

	if (yMax > image.size().height) {
		yMaxPad = yMax - image.size().height;
	}
	else {
		yMaxPad = 0;
	}

	//compute the acual rectangle we will extract from the image
	cv::Rect extractionRegion = cv::Rect(xMin + xMinPad,
		yMin + yMinPad,
		(xMax - xMin) - xMaxPad - xMinPad,
		(yMax - yMin) - yMaxPad - yMinPad);

	//make sure the patch is not completely outside the image
	if (extractionRegion.x + extractionRegion.width > 0 &&
		extractionRegion.y + extractionRegion.height > 0 &&
		extractionRegion.x < image.cols &&
		extractionRegion.y < image.rows) {

		cv::Mat real_patch(region.size, image.type());


		//replicate along borders if needed
		if (xMinPad > 0 || xMaxPad > 0 || yMinPad > 0 || yMaxPad > 0) {
			cv::copyMakeBorder(image(extractionRegion), real_patch, yMinPad,
				yMaxPad, xMinPad, xMaxPad, cv::BORDER_REPLICATE);

		}
		else {
			real_patch = image(extractionRegion);
		}

		if (!(real_patch.size().width == region.size.width && real_patch.size().height == region.size.height)) {
			//cout << "kasst" << endl;
		}

		cv::Mat ds_patch;
		cv::resize(real_patch, ds_patch, output_sz);

		return ds_patch;

	}
	else {
		cv::Mat dummyRegion = cv::Mat::zeros(region.size, image.type());
		cv::Mat ds_patch;
		cv::resize(dummyRegion, ds_patch, output_sz);

		return ds_patch;
	}
}
void BACF::update_impl(const cv::Mat& image, const TrackedRegion& region, const float frame) {
	//extract pixels to use for update
	cv::Mat pixels = extractTrackedRegion(image, region, p.model_sz);
	std::vector<cv::Mat> feature_vecf = compute_feature_vec(pixels);

	if (frame == 1.0) {
		model_xf = feature_vecf;
	}
	else {
		for (int i = 0; i < model_xf.size(); i++)
			model_xf[i] = ((1 - p.update_rate) * model_xf[i]) + (p.update_rate * feature_vecf[i]);
	}

	compute_ADMM();
}

cv::Mat BACF::channelMultiply(std::vector<cv::Mat> a, std::vector<cv::Mat> b, int flags, bool conjb) {
	//CV_Assert(a.size() == b.size());

	cv::Mat prod;
	cv::Mat sum = cv::Mat::zeros(a[0].size(), a[0].type());
	for (unsigned int i = 0; i < a.size(); ++i) {
		cv::Mat ca = a[i];
		cv::Mat cb = b[i];
		cv::mulSpectrums(a[i], b[i], prod, flags, conjb);
		sum += prod;
	}
	return sum;
}

void BACF::detect(const cv::Mat& image) {
	cv::Mat response = detect_impl(image, target);
	//	cout << response;

	//resp_newton
	std::pair<int, int> maxpos = minMaxLoc(response);

	cv::Point2i translation(round(shift_index(maxpos.first, response.cols) * scale_factor),
		round(shift_index(maxpos.second, response.rows) * scale_factor));

	target.center = target.center + translation;
}

cv::Mat BACF::detect_impl(const cv::Mat& image, const TrackedRegion& region) {
	cv::Mat pixels = extractTrackedRegion(image, region, p.model_sz);

	std::vector<cv::Mat> feature_vecf = compute_feature_vec(pixels);

	return compute_response(filterf, feature_vecf);
}

std::vector<cv::Mat> BACF::fft2(const cv::Mat featureData) {
	std::vector<cv::Mat> channels(featureData.channels());
	std::vector<cv::Mat> channelsf(featureData.channels());
	cv::split(featureData, channels);

	for (size_t i = 0; i < channels.size(); ++i) {
		cv::Mat windowed;
		multiply(channels[i], window, windowed);
		cv::dft(windowed, channelsf[i], cv::DFT_REAL_OUTPUT);
	}

	return channelsf;
}

int BACF::shift_index(const int index, const int length) const
{
	return (index > length / 2) ? -length + index : index;
}

cv::Mat BACF::make_labels(const cv::Size matrix_size, const cv::Size target_size, const float sigma_factor) const {
	cv::Mat new_labels(matrix_size.height, matrix_size.width, CV_32F);

	const float sigma = std::sqrt((float)target_size.area()) * sigma_factor;
	const float constant = -0.5 / pow(sigma, 2);

	for (int x = 0; x < matrix_size.width; x++) {
		for (int y = 0; y < matrix_size.height; y++) {
			int shift_x = shift_index(x, matrix_size.width);
			int shift_y = shift_index(y, matrix_size.height);
			float value = std::exp(constant * (std::pow(shift_x, 2) + std::pow(shift_y, 2)));
			new_labels.at<float>(y, x) = value;
		}
	}

	cv::Mat labels_dft;

	cv::dft(new_labels, labels_dft, cv::DFT_REAL_OUTPUT);

	return labels_dft;
}

void BACF::compute_ADMM() {
	std::vector<cv::Mat>l_f;
	std::vector<cv::Mat>h_f;
	int mu = 1;
	float T = (float)p.model_sz.area();
	cv::Mat S_xx = channelMultiply(model_xf, model_xf, 0, true);
	filterf.clear();
	for (int i = 0; i < 3; i++)
	{
		l_f.emplace_back(cv::Mat::zeros(model_xf[0].size(), model_xf[0].type()));
		h_f.emplace_back(cv::Mat::zeros(model_xf[0].size(), model_xf[0].type()));
		filterf.emplace_back(cv::Mat::zeros(model_xf[0].size(), model_xf[0].type()));
	}
	for (int i = 0; i < 2; i++)
	{
		cv::Mat B = S_xx + (T * mu);
		cv::Mat S_lx = channelMultiply(l_f, model_xf, 0, true);
		cv::Mat S_hx = channelMultiply(h_f, model_xf, 0, true);
		for (int j = 0; j < model_xf.size(); j++)
		{
			cv::Mat mlabelf, S_xxyf, mS_lx, mS_hx;
			multiply(labelsf, model_xf[j], mlabelf);
			multiply(S_xx, mlabelf, S_xxyf);
			multiply(S_lx, model_xf[j], mS_lx);
			multiply(S_hx, model_xf[j], mS_hx);
			cv::Mat h;
			cv::Mat ghj;
			divide(S_xxyf.mul(1 / (T * mu)) - mS_lx.mul(1 / mu) + mS_hx, B, ghj);
			filterf[j] = (mlabelf.mul(1 / (T * mu)) - l_f[j].mul(1 / mu) + h_f[j]) - ghj;
			cv::dft((filterf[j].mul(mu) + l_f[j]), h, cv::DFT_INVERSE | cv::DFT_SCALE| cv::DFT_REAL_OUTPUT);
			cv::Mat t = extractTrackedRegionSpec(h.mul(1 / mu), p.model_sz);
			cv::dft(t, h_f[j], cv::DFT_REAL_OUTPUT);
			l_f[j] = l_f[j] + ((filterf[j].mul(mu) - h_f[j]));
		}
		mu = 10;
	}
}

cv::Mat BACF::extractTrackedRegionSpec(cv::Mat model, const cv::Size output_sz)
{
	cv::Mat lp = model.clone();
	for (int x = 0; x < output_sz.width; x++)
	{
		for (int y = 0; y < output_sz.height; y++)
		{
			if ((x > ceil(output_sz.width) / 4 && x < ceil(output_sz.width / 4) + output_sz.width / 2) && (y > ceil(output_sz.height / 4) && y < ceil(output_sz.height / 4) + output_sz.height / 2))
				lp.at<float>(y, x) = model.at<float>(y, x);
			else
				lp.at<float>(y, x) = 0;
		}
	}
	return lp;
}

cv::Mat BACF::compute_response(const std::vector<cv::Mat>& filter, const std::vector<cv::Mat>& sample)
{
	cv::Mat response;

	cv::Mat resp_dft = channelMultiply(filter, sample, 0, false);
	cv::dft(resp_dft, response, cv::DFT_INVERSE | cv::DFT_SCALE| cv::DFT_REAL_OUTPUT);
	// cout << response;
	return response;
}

class BoxExtractor
{
public:
	cv::Rect2d extract(cv::Mat img);
	cv::Rect2d extract(const std::string& windowName, cv::Mat img, bool showCrossair = true);

	struct handlerT
	{
		bool isDrawing;
		cv::Rect2d box;
		cv::Mat image;

		// initializer list
		handlerT() : isDrawing(false) {};
	} params;

private:
	static void mouseHandler(int event, int x, int y, int flags, void* param);
	void opencv_mouse_callback(int event, int x, int y, int, void* param);
};

int main(int argc, char** argv)
{

	// ROI selector
	BoxExtractor box;

	// create the tracker

	// set input video
	std::string video = "K:\\ReCFjs\\wakeboard7\\img\\";
	// VideoCapture cap(video);

	cv::Mat frame = cv::imread(video + "000001.jpg");
	if (frame.empty())
	{
		std::cout << "first frame is empty" << std::endl;
		return 0;
	}

	// get bounding box
	// cap >> frame;
	cv::Rect2d roi = cv::Rect2d(623, 299, 11, 38);

	// quit if ROI was not selected
	if (roi.width == 0 || roi.height == 0)
		return 0;

	BACF* tracker = new BACF();
	tracker->initialize(frame, roi);

	// do the tracking
	printf("Start the tracking process, press ESC to quit.\n");
	for (const auto& entry : std::filesystem::directory_iterator(video))
	{
		std::cout << entry.path() << std::endl;
		frame = cv::imread(entry.path().string());
		if (frame.empty())
			break;
		// stop the program if no more images
		if (frame.rows == 0 || frame.cols == 0)
			break;

		// update the tracking result
		tracker->updateImpl(frame, roi);

		// draw the tracked object
		rectangle(frame, roi, cv::Scalar(255, 0, 0), 2, 1);

		// show image with the tracked object
		imshow("tracker", frame);

		// quit on ESC button
		if (cv::waitKey(1) == 27)
			break;
	}
}

void BoxExtractor::mouseHandler(int event, int x, int y, int flags, void* param)
{
	BoxExtractor* self = static_cast<BoxExtractor*>(param);
	self->opencv_mouse_callback(event, x, y, flags, param);
}

void BoxExtractor::opencv_mouse_callback(int event, int x, int y, int, void* param)
{
	handlerT* data = (handlerT*)param;
	switch (event)
	{
		// update the selected bounding box
	case cv::EVENT_MOUSEMOVE:
		if (data->isDrawing)
		{
			data->box.width = x - data->box.x;
			data->box.height = y - data->box.y;
		}
		break;

		// start to select the bounding box
	case cv::EVENT_LBUTTONDOWN:
		data->isDrawing = true;
		data->box = cv::Rect(x, y, 0, 0);
		break;

		// cleaning up the selected bounding box
	case cv::EVENT_LBUTTONUP:
		data->isDrawing = false;
		if (data->box.width < 0)
		{
			data->box.x += data->box.width;
			data->box.width *= -1;
		}
		if (data->box.height < 0)
		{
			data->box.y += data->box.height;
			data->box.height *= -1;
		}
		break;
	}
}

cv::Rect2d BoxExtractor::extract(cv::Mat img)
{
	return extract("Bounding Box Extractor", img);
}

cv::Rect2d BoxExtractor::extract(const std::string& windowName, cv::Mat img, bool showCrossair)
{

	int key = 0;

	// show the image and give feedback to user
	imshow(windowName, img);
	printf("Select an object to track and then press SPACE/BACKSPACE/ENTER button!\n");

	// copy the data, rectangle should be drawn in the fresh image
	params.image = img.clone();

	// select the object
	cv::setMouseCallback(windowName, mouseHandler, (void*)&params);

	// end selection process on SPACE (32) BACKSPACE (27) or ENTER (13)
	while (!(key == 32 || key == 27 || key == 13))
	{
		// draw the selected object
		rectangle(
			params.image,
			params.box,
			cv::Scalar(255, 0, 0), 2, 1);

		// draw cross air in the middle of bounding box
		if (showCrossair)
		{
			// horizontal line
			line(
				params.image,
				cv::Point((int)params.box.x, (int)(params.box.y + params.box.height / 2)),
				cv::Point((int)(params.box.x + params.box.width), (int)(params.box.y + params.box.height / 2)),
				cv::Scalar(255, 0, 0), 2, 1);

			// vertical line
			line(
				params.image,
				cv::Point((int)(params.box.x + params.box.width / 2), (int)params.box.y),
				cv::Point((int)(params.box.x + params.box.width / 2), (int)(params.box.y + params.box.height)),
				cv::Scalar(255, 0, 0), 2, 1);
		}

		// show the image bouding box
		imshow(windowName, params.image);

		// reset the image
		params.image = img.clone();

		// get keyboard event
		key = cv::waitKey(1);
	}

	return params.box;
}
