#ifndef PLACE_NUMBER_RECOGNIZER_H
#define PLACE_NUMBER_RECOGNIZER_H

#include <string>

#include <opencv2/opencv.hpp>


typedef cv::Vec<unsigned char, 3> Color;

std::string recognize_place_number(const cv::Mat &image, const Color &color,
                                   const cv::Rect &search_rect);


#endif // PLACE_NUMBER_RECOGNIZER_H
