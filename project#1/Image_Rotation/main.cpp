#include <iostream>
#include <opencv2\opencv.hpp>

cv::Mat problem_a_rotate_forward(cv::Mat img, double angle){
	cv::Mat output;
	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	//Define Affine Matrix for forward method
	cv::Mat affmat = cv::Mat::zeros(2, 2, CV_32F);
	float* aff_p = (float*) affmat.data;
	double rad_angle = angle * CV_PI / 180;
	aff_p[0] = cos(rad_angle);	aff_p[1] = -sin(rad_angle);
	aff_p[2] = sin(rad_angle);	aff_p[3] = cos(rad_angle);

	//Set the size of the output image
	//Left Upper Point														
	int lup[2] = { 0,0 };
	//Left Lower Point
	int llp[2] = { int(round(cos(rad_angle) * img.rows)), int(round(sin(rad_angle) * img.rows)) };
	//Right Upper Point																				
	int rup[2] = { int(round(-sin(rad_angle) * img.cols)), int(round(cos(rad_angle) * img.cols)) };
	//Right Lower Point
	int rlp[2] = { int(round(cos(rad_angle) * img.rows - sin(rad_angle) * img.cols)), int(round(sin(rad_angle) * img.rows + cos(rad_angle) * img.cols)) };

	int max_x = int(round(std::max({ lup[0], llp[0], rup[0], rlp[0] })));
	int min_x = int(round(std::min({ lup[0], llp[0], rup[0], rlp[0] })));
	int max_y = int(round(std::max({ lup[1], llp[1], rup[1], rlp[1] })));
	int min_y = int(round(std::min({ lup[1], llp[1], rup[1], rlp[1] })));
	int img_rows = max_x - min_x + 1;
	int img_cols = max_y - min_y + 1;
	output = cv::Mat::zeros(img_rows, img_cols, CV_8UC3); // img size : img_rows X img_cols
	unsigned* output_p = (unsigned*)output.data;
	unsigned* img_p = (unsigned*)img.data;

	//Mapping
	cv::Mat pixel = cv::Mat::zeros(2, 1, CV_32F);	float* px = (float*)pixel.data;	// (x,y)
	cv::Mat pixel_m = cv::Mat::zeros(2, 1, CV_32F); float* px_m = (float*)pixel_m.data; // (x',y')
	for (int r = 0; r < img.rows; r++) {
		for (int c = 0; c < img.cols; c++) {
			px[0] = r;
			px[1] = c;
			pixel_m = affmat * pixel;
			for (int color = 0; color < 3; color++) {
				output.at<cv::Vec3b>(round(px_m[0] - min_x), round(px_m[1] - min_y))[color] = img.at<cv::Vec3b>(px[0], px[1])[color];
			}
		}
	}
	
	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////
	cv::imshow("a_output", output); cv::waitKey(0);
	return output;
}

cv::Mat problem_b_rotate_backward(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	//Define Affine Matrix for forward method
	cv::Mat affmat = cv::Mat::zeros(2, 2, CV_32F);
	float* aff_p = (float*)affmat.data;
	double rad_angle = angle * CV_PI / 180;
	aff_p[0] = cos(rad_angle);	aff_p[1] = -sin(rad_angle);
	aff_p[2] = sin(rad_angle);	aff_p[3] = cos(rad_angle);
	
	//Define Inverse Affine Matrix for Backward method
	cv::Mat invmat = affmat.inv();

	//Set the size of the output image
	//Left Upper Point														
	int lup[2] = { 0,0 };
	//Left Lower Point
	int llp[2] = { int(round(cos(rad_angle) * img.rows)), int(round(sin(rad_angle) * img.rows)) };
	//Right Upper Point																				
	int rup[2] = { int(round(-sin(rad_angle) * img.cols)), int(round(cos(rad_angle) * img.cols)) };
	//Right Lower Point
	int rlp[2] = { int(round(cos(rad_angle) * img.rows - sin(rad_angle) * img.cols)), int(round(sin(rad_angle) * img.rows + cos(rad_angle) * img.cols)) };

	int max_x = int(round(std::max({ lup[0], llp[0], rup[0], rlp[0] })));
	int min_x = int(round(std::min({ lup[0], llp[0], rup[0], rlp[0] })));
	int max_y = int(round(std::max({ lup[1], llp[1], rup[1], rlp[1] })));
	int min_y = int(round(std::min({ lup[1], llp[1], rup[1], rlp[1] })));
	int img_rows = max_x - min_x + 1;
	int img_cols = max_y - min_y + 1;
	output = cv::Mat::zeros(img_rows, img_cols, CV_8UC3); // img size : img_rows X img_cols
	unsigned* output_p = (unsigned*)output.data;
	unsigned* img_p = (unsigned*)img.data;

	//Mapping
	cv::Mat pixel = cv::Mat::zeros(2, 1, CV_32F);	float* px = (float*)pixel.data;	// (x,y)
	cv::Mat pixel_m = cv::Mat::zeros(2, 1, CV_32F); float* px_m = (float*)pixel_m.data; // (x',y')
	for (int r = 0; r < output.rows; r++) {
		for (int c = 0; c < output.cols; c++) {
			px_m[0] = r + min_x;
			px_m[1] = c + min_y;
			pixel = invmat * pixel_m;
			for (int color = 0; color < 3; color++) {
				if (round(px[0]) >= 0 && round(px[0]) < img.rows && round(px[1]) >= 0 && round(px[1]) < img.cols) {
					output.at<cv::Vec3b>(px_m[0] - min_x, px_m[1] - min_y)[color] = img.at<cv::Vec3b>(round(px[0]), round(px[1]))[color];
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("b_output", output); cv::waitKey(0);

	return output;
}

cv::Mat problem_c_rotate_backward_interarea(cv::Mat img, double angle){
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	//Define Affine Matrix for forward method
	cv::Mat affmat = cv::Mat::zeros(2, 2, CV_32F);
	float* aff_p = (float*)affmat.data;
	double rad_angle = angle * CV_PI / 180;
	aff_p[0] = cos(rad_angle);	aff_p[1] = -sin(rad_angle);
	aff_p[2] = sin(rad_angle);	aff_p[3] = cos(rad_angle);

	//Define Inverse Affine Matrix for Backward method
	cv::Mat invmat = affmat.inv();

	//Set the size of the output image
	//Left Upper Point														
	int lup[2] = { 0,0 };
	//Left Lower Point
	int llp[2] = { int(round(cos(rad_angle) * img.rows)), int(round(sin(rad_angle) * img.rows)) };
	//Right Upper Point																				
	int rup[2] = { int(round(-sin(rad_angle) * img.cols)), int(round(cos(rad_angle) * img.cols)) };
	//Right Lower Point
	int rlp[2] = { int(round(cos(rad_angle) * img.rows - sin(rad_angle) * img.cols)), int(round(sin(rad_angle) * img.rows + cos(rad_angle) * img.cols)) };

	int max_x = int(round(std::max({ lup[0], llp[0], rup[0], rlp[0] })));
	int min_x = int(round(std::min({ lup[0], llp[0], rup[0], rlp[0] })));
	int max_y = int(round(std::max({ lup[1], llp[1], rup[1], rlp[1] })));
	int min_y = int(round(std::min({ lup[1], llp[1], rup[1], rlp[1] })));
	int img_rows = max_x - min_x + 1;
	int img_cols = max_y - min_y + 1;
	output = cv::Mat::zeros(img_rows, img_cols, CV_8UC3); // img size : img_rows X img_cols
	unsigned* output_p = (unsigned*)output.data;
	unsigned* img_p = (unsigned*)img.data;

	//Mapping
	int A, B, C, D; unsigned K; float x, p, M, N; // Variables For Interpolation
	cv::Mat pixel = cv::Mat::zeros(2, 1, CV_32F);	float* px = (float*)pixel.data;	// (x,y)
	cv::Mat pixel_m = cv::Mat::zeros(2, 1, CV_32F); float* px_m = (float*)pixel_m.data; // (x',y')
	for (int r = 0; r < output.rows; r++) {
		for (int c = 0; c < output.cols; c++) {
			px_m[0] = r + min_x;
			px_m[1] = c + min_y;
			pixel = invmat * pixel_m;

			for (int color = 0; color < 3; color++) {
				if (floor(px[0]) >= 0 && ceil(px[0]) < img.rows && floor(px[1]) >= 0 && ceil(px[1]) < img.cols) {
					B = img.at<cv::Vec3b>(floor(px[0]), floor(px[1]))[color];
					C = img.at<cv::Vec3b>(floor(px[0]), ceil(px[1]))[color];
					A = img.at<cv::Vec3b>(ceil(px[0]), floor(px[1]))[color];
					D = img.at<cv::Vec3b>(ceil(px[0]), ceil(px[1]))[color];
					x = ceil(px[0]) - px[0];
					p = ceil(px[1]) - px[1];
					M = A * (1 - x) + B * x;
					N = D * (1 - x) + C * x;
					K = round(p * M + (1 - p) * N);
					output.at<cv::Vec3b>(px_m[0] - min_x, px_m[1] - min_y)[color] = K;
				}
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("c_output", output); cv::waitKey(0);

	return output;
}

cv::Mat Example_change_brightness(cv::Mat img, int num, int x, int y) {
	/*
	img : input image
	num : number for brightness (increase or decrease)
	x : x coordinate of image (for square part)
	y : y coordinate of image (for square part)

	*/
	cv::Mat output = img.clone();
	int size = 100;
	int height = (y + 100 > img.cols) ? img.cols : y + 100;
	int width = (x + 100 > img.rows) ? img.rows : x + 100;

	for (int i = x; i < width; i++)
	{
		for (int j = y; j < height; j++)
		{
			for (int c = 0; c < img.channels(); c++)
			{
				int t = img.at<cv::Vec3b>(i, j)[c] + num;
				output.at<cv::Vec3b>(i, j)[c] = t > 255 ? 255 : t < 0 ? 0 : t;
			}
		}

	}
	cv::imshow("output1", img);
	cv::imshow("output2", output);
	cv::waitKey(0);
	return output;
}

int main(void){

	double angle = -15.0f;

	cv::Mat input = cv::imread("lena.jpg");
	//Fill problem_a_rotate_forward and show output
	problem_a_rotate_forward(input, angle);
	//Fill problem_b_rotate_backward and show output
	problem_b_rotate_backward(input, angle);
	//Fill problem_c_rotate_backward_interarea and show output
	problem_c_rotate_backward_interarea(input, angle);
	//Example how to access pixel value, change params if you want
	Example_change_brightness(input, 100, 50, 125);
}