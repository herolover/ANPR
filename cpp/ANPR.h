#ifndef ANPR_H
#define ANPR_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>


class ANPR
{
public:
  void set_image(const cv::Mat &image);

  void run();

  cv::Mat get_number_plate_image() const;
  std::string get_number_plate_text() const;
private:
  cv::Mat preprocess_image(const cv::Mat &image) const;

  enum EdgeType
  {
    ET_VERTICAL,
    ET_HORIZONTAL
  };

  cv::Mat compute_edge_image(const cv::Mat &image, EdgeType edge_type) const;

  cv::Mat compute_skew_correction_matrix(const cv::Mat &image) const;

  std::vector<cv::Rect> find_contrast_rects(const cv::Mat &image, int margin) const;

  std::string recognize_text(const cv::Mat &image) const;


  cv::Mat image_;
  std::string number_plate_text_;
  cv::Mat number_plate_image_;
};


#endif // ANPR_H
