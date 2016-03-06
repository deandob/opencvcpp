#include "opencv2/opencv.hpp"
#include "JSON.h"
#include "JSONValue.h"
#include <iostream>
#include <string>

// Compile settings: c++ general include directories: C:\Users\Dean\Downloads\opencv\opencv\build\include
// Linker settings: gbeneral libraries: C:\Users\Dean\Downloads\opencv\opencv\build\x64\vc12\lib
// Linker settings: input additional dependencies: C:\Users\Dean\Downloads\opencv\opencv\build\x64\vc12\lib

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {				// #1 = display (1/0), #2 = diff Change, #3 = blursize, #4 = blurSD, #5 = frame rate, #6 = rtsp string, #7 = JSON mask array

	if (argc < 8) {
		std::cerr << "ERROR: Not enough arguments. Use display bool, diff change int, blur size int, blur SD int, rtsp string, JSON mask array string." << std::endl;
		return -1;
	}

	int frameRate = atol(argv[5]) + 1;			// add 1 to the framerate to catch up startup 
	int lpWait = int(1000 / frameRate);			// process loop delay
	int diffChange = atol(argv[2]);				//amount of the change between frame pixels before registering as a change
	int blurSize = atol(argv[3]);				// Apply gaussian blur to remove noise, creates large pixels
	int blurSD = atol(argv[4]);					// smooth out the change between the large pixel

	cv::VideoCapture vcap;
	cv::Mat frameMat;

	int camNum = 0;

	bool display = (atol(argv[1]) == 1);
	string rtsp = argv[6];
	unsigned long matSum;

	// cerr used for general message outputs, cout used to communicate to node.js via pipe
	std::cerr << "Connecting to camera - " << rtsp << std::endl;

	//open the video stream and make sure it's opened
	if (!vcap.open(rtsp)) {
		std::cerr << "Error opening video stream or file" << std::endl;
		return -1;
	}

	// Size the mask matrix based on size of video frame
	vcap >> frameMat;
	cv::Mat processedMat = Mat::zeros(frameMat.rows, frameMat.cols, CV_8UC1);
	cv::Mat maskMat = Mat::zeros(frameMat.rows, frameMat.cols, CV_8UC1);
	cv::Mat previousMat = Mat::zeros(frameMat.rows, frameMat.cols, CV_8UC1);
	cv::Mat diffMat = Mat::zeros(frameMat.rows, frameMat.cols, CV_8UC1);
	cv::Mat threshMat = Mat::zeros(frameMat.rows, frameMat.cols, CV_8UC1);
	cv::Mat maskedMat = Mat::zeros(frameMat.rows, frameMat.cols, CV_8UC1);
	//cv::Mat kernelMat = Mat::ones(10, 10, CV_8UC1);

	std::vector<std::vector<cv::Point> > contours;

	//JSONValue *value = JSON::Parse("[{\"x1\":0.07468354430379746,\"x2\":0.8658227848101265,\"y1\":0.15569620253164557,\"y2\":0.47848101265822784}]");
	JSONValue *value = JSON::Parse(argv[7]);
	if (value != NULL) {
		JSONArray root;
		if (value->IsArray() == true) {
			JSONArray masks = value->AsArray();
			for (unsigned int i = 0; i < masks.size(); i++)
			{
				JSONObject mask = masks[i]->AsObject();
				std::cerr << "Setting mask #" << std::to_string(i) << 
					" x1:" << std::to_string(mask[L"x1"]->AsNumber()) << 
					", x2:" << std::to_string(mask[L"x2"]->AsNumber()) << 
					", y1:" << std::to_string(mask[L"y1"]->AsNumber()) << 
					", y2:" << std::to_string(mask[L"y2"]->AsNumber()) << 
					std::endl;
				for (unsigned int j = 0; j < mask.size(); j++)
				{
					maskMat.rowRange(int(mask[L"y1"]->AsNumber() * maskMat.rows), int(mask[L"y2"]->AsNumber() * maskMat.rows))    // saved mask is proportional to image width
						.colRange(int(mask[L"x1"]->AsNumber() * maskMat.cols), int(mask[L"x2"]->AsNumber() * maskMat.cols))
						.setTo(Scalar(255));
				}
			}
		}
		else {
			std::cerr << "Mask is not a JSON array. Exiting." << std::endl;
			return -1;
		}
	}
	else {
		std::cerr << "No Mask supplied. Motion detection will be performed on entire image." << std::endl;
		maskedMat = Mat::ones(frameMat.rows, frameMat.cols, CV_8UC1);
	}

	if (display) {
		cv::namedWindow("Input Window");
		cv::namedWindow("Output Window");
	}

	int maxChange = int(cv::sum(maskMat)[0] / 255);      // max change possible if all pixels in the mast are different than last frame
	std::cout << "Max:" + std::to_string(camNum) + "," + std::to_string(maxChange) << std::endl;

	for (;;) {
		vcap >> frameMat;
		cv::cvtColor(frameMat, processedMat, CV_BGR2GRAY);											// remove colours
		cv::GaussianBlur(processedMat, processedMat, Size(blurSize, blurSize), blurSD, blurSD);		// add blur to remove small object motion
		processedMat.copyTo(maskedMat, maskMat);              // this sets the mask
		//cv::bitwise_and(processedMat, maskMat, maskedMat);
		cv::absdiff(maskedMat, previousMat, diffMat);
		maskedMat.copyTo(previousMat);
		//cv::bitwise_and(processedMat, maskMat, previousMat);        
		cv::threshold(diffMat, threshMat, diffChange, 255, CV_THRESH_BINARY);         // discard small changes & make all changed pixels 255 (x, y, z, a, b) z = threshold val make 'a' for any change over this threshold
		cv::dilate(threshMat, threshMat, Mat(), Point(-1, -1), 2, 1, 1);				// join gaps in a contour (where lighter areas previously have been removed for noise threshold

		if (display) {
			cv::imshow("Input Window", frameMat);
			//cv::imshow("Input Window", processedMat);
			cv::imshow("Output Window", threshMat);
		}
		matSum = (unsigned long)(cv::sum(threshMat)[0] / 255);
		cv::findContours(threshMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);		// number of blobs
		//std::cout << "Diff:" + std::to_string(camNum) + "," + std::to_string(matSum) + "," + std::to_string(contours.size()) << std::endl;
		std::cout << "Diff:" + std::to_string(camNum) + "," + std::to_string(matSum);
		for (int j = 0; j < contours.size(); ++j) {												// output the end point of all vectors in contours to calculate distance moved
			std::cout << "," + std::to_string(contours[0][j].x) + "," + std::to_string(contours[0][j].y);
		}
		std::cout << std::endl;
		if (cv::waitKey(lpWait) >= 0) break;							// wait for delay before processing a new frame, or exit if a key is pressed
	}
	std::cerr << "Exiting motion detection due to keypress." << std::endl;
}

//if (matSum > largeChangeThresh * maxChange) console.log("large change")
//    //if (display) console.log("Motion val: " + matSum + " (trigger: " + normTrigger + ", tooLarge: " + largeChangeThresh * maxChange + ")")
//    if (matSum > normTrigger && matSum < largeChangeThresh * maxChange) {
//        alarmTriggeredTimer = 0;                                                    // wait until motion stopped before letting timer count for another trigger
//        primeTrigTimer = primeTrigTimer + 1                                         // add up number of consequetive changed frames
//        if (cameras[cam].alarmed === false && primeTrigTimer > frameRate * motionDurationToTrig) {
//            primeTrigTimer = 0;
//            cameras[cam].alarmed = true;
//            motionAlarm.emit("alarm", camNum, matSum)
//        }
//    } else {
//        primeTrigTimer = 0;
//    }
//    if (cameras[cam].alarmed === true) {
//        alarmTriggeredTimer = alarmTriggeredTimer + 1;                              // wait until releaseTrigTime before resetting motion alarm
//        if (alarmTriggeredTimer > frameRate * releaseTrigTime) {
//            cameras[cam].alarmed = false;
//        }
//    }
