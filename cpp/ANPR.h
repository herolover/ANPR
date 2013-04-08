#ifndef ANPR_H
#define ANPR_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>


class ANPR
{
public:
  void set_image(const cv::Mat &image);
  void set_image(const cv::Mat &image, const cv::Rect &search_rect);

  void find_and_recognize();

  cv::Mat get_number_plate_image() const;
  std::string get_number_plate_text() const;

private:
  cv::Rect find_number_plate_rect(const cv::Mat &image) const;
  void recognize_text();


  cv::Rect search_rect_;
  cv::Mat image_;
  std::string number_plate_text_;
  cv::Mat number_plate_image_;
};


#endif // ANPR_H
