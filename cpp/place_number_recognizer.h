#ifndef PLACE_NUMBER_RECOGNIZER_H
#define PLACE_NUMBER_RECOGNIZER_H

#include <string>
#include <utility>

#include <opencv2/opencv.hpp>


typedef cv::Vec<unsigned char, 3> Color;

std::pair<std::string, cv::Rect> recognize_place_number(const cv::Mat &image,
                                                        const Color &color,
                                                        const cv::Rect &search_rect);


#endif // PLACE_NUMBER_RECOGNIZER_H
