#ifndef PLACE_NUMBER_RECOGNIZER_H
#define PLACE_NUMBER_RECOGNIZER_H

#include <string>

#include <opencv2/opencv.hpp>


typedef cv::Vec<unsigned char, 3> Color;

class PlaceNumberRecognizer
{
public:
  void set_image(const cv::Mat &image);
  void set_color(const Color &color);
  std::string get_place_number() const;

  void find_and_recognize();
private:
  cv::Mat image_;
  Color color_;
  std::string place_number_;
};


#endif // PLACE_NUMBER_RECOGNIZER_H
