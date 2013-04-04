#include "ANPR.h"

#include <string>

#include <boost/format.hpp>
#include <boost/regex.hpp>

#include <tesseract/baseapi.h>

#include "help_alg.h"
#include "help_opencv.h"


#include <stdio.h>


void ANPR::set_image(const cv::Mat &image)
{
  this->image_ = image.clone();

  int img_top_offset = this->image_.rows * 1 / 4;
  int img_bottom_offset = this->image_.rows * 1 / 9;
  int img_left_offset = this->image_.cols / 8;
  int img_right_offset = this->image_.cols / 8;

  this->search_rect_.x = img_left_offset;
  this->search_rect_.y = img_top_offset;
  this->search_rect_.width = this->image_.cols - img_right_offset - img_left_offset;
  this->search_rect_.height = this->image_.rows - img_bottom_offset - img_top_offset;
}


void ANPR::set_image(const cv::Mat &image, const cv::Rect &search_rect)
{
  this->image_ = image.clone();
  this->search_rect_ = search_rect;
}


void ANPR::find_and_recognize()
{
  double angle = compute_skew_correction_angle(this->image_(this->search_rect_));

  cv::Mat skew_matrix = cv::Mat::eye(2, 3, CV_64FC1);
  skew_matrix.at<double>(1, 0) = tan(CV_PI * 0.5 - angle);
  skew_matrix.at<double>(1, 2) = -(this->search_rect_.x +
                                   this->search_rect_.width * 0.5) / tan(angle);

  cv::Mat deskewed_image;
  cv::warpAffine(this->image_, deskewed_image, skew_matrix, this->image_.size());

  auto rects = this->find_contrast_rects(deskewed_image, 1);
  if (rects.size() > 0)
  {
    this->number_plate_image_ = deskewed_image(rects[0]);
    this->recognize_text();
  }
}


cv::Mat ANPR::get_number_plate_image() const
{
  return this->number_plate_image_;
}


std::string ANPR::get_number_plate_text() const
{
  return this->number_plate_text_;
}


std::vector<cv::Rect> ANPR::find_contrast_rects(const cv::Mat &image,
                                                int margin) const
{
  int width = 640;
  double ratio = (double)image.cols / width;
  cv::Size2f size(width, image.rows / ratio);

  cv::Mat small_image;
  cv::resize(image, small_image, size);

  cv::Rect search_rect = this->search_rect_ * (1.0 / ratio);
  cv::Mat cropped_image = small_image(search_rect);

  cv::Mat proc_image = convert_to_grayscale_and_remove_noise(cropped_image);
  cv::Mat vertical_edge_image = compute_edge_image(proc_image, ET_VERTICAL);

  std::vector<double> rows_rms_contrast;
  for (int i = 0; i < vertical_edge_image.rows; ++i)
    rows_rms_contrast.push_back(row_RMS(vertical_edge_image.row(i)));

  rows_rms_contrast = simple_moving_average(rows_rms_contrast, 10, 0.0);

  save_to_file("rows", rows_rms_contrast.begin(), rows_rms_contrast.end());


  auto rows_rms_contrast_pairs = find_local_pairs(rows_rms_contrast.begin(),
                                                  rows_rms_contrast.end(),
                                                  0.5);

  std::vector<cv::Rect> rects;
  for (unsigned i = 0; i < rows_rms_contrast_pairs.size(); ++i)
  {
    int top_bound = std::distance(rows_rms_contrast.begin(),
                                  rows_rms_contrast_pairs[i].first) - margin;
    int bottom_bound = std::distance(rows_rms_contrast.begin(),
                                     rows_rms_contrast_pairs[i].second) + margin;

    if (top_bound < 0)
      top_bound = 0;
    if (bottom_bound >= (int)rows_rms_contrast.size())
      bottom_bound = rows_rms_contrast.size();

    cv::Mat both_edge_image = vertical_edge_image +
                              compute_edge_image(proc_image, ET_HORIZONTAL);

    cv::Mat rows_range = both_edge_image.rowRange(top_bound, bottom_bound);

    std::vector<double> cols_rms_contrast;
    for (int j = 0; j < rows_range.cols; ++j)
      cols_rms_contrast.push_back(col_RMS(rows_range.col(j)));

    cols_rms_contrast = simple_moving_average(cols_rms_contrast, 30, 0.0);

    save_to_file(boost::str(boost::format("cols_%1%") % i),
                 cols_rms_contrast.begin(), cols_rms_contrast.end());


    auto cols_rms_contrast_pairs = find_local_pairs(cols_rms_contrast.begin(),
                                                    cols_rms_contrast.end(),
                                                    0.3);

    for (auto &cols_pair: cols_rms_contrast_pairs)
    {
      int left_bound = std::distance(cols_rms_contrast.begin(), cols_pair.first) - margin;
      int right_bound = std::distance(cols_rms_contrast.begin(), cols_pair.second) + margin;

      cv::Rect rect;

      rect.x = left_bound + search_rect.x;
      rect.y = top_bound + search_rect.y;
      rect.height = bottom_bound - top_bound;
      rect.width = right_bound - left_bound;

      rects.push_back(rect * ratio);
    }
  }

  return rects;
}


void ANPR::recognize_text()
{
  cv::Mat proc_image = convert_to_grayscale_and_remove_noise(this->number_plate_image_);

  cv::Mat threshold_image;
  cv::adaptiveThreshold(proc_image, threshold_image, 255.0,
                        CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV,
                        151, 0.0);


  // fill a small noise on a characters
  std::vector<std::vector<cv::Point> > noisy_areas = find_filled_areas(threshold_image.clone(), 0);
  for (auto &area: noisy_areas)
    if (area.size() < 60)
      draw_area(threshold_image, area, 255);


  std::vector<std::vector<cv::Point> > areas = find_filled_areas(threshold_image.clone(), 255);

  // remove too wide or too small areas
  areas.erase(std::remove_if(areas.begin(), areas.end(),
                             [threshold_image](const std::vector<cv::Point> &area)
  {
    cv::Rect area_bound = cv::boundingRect(area);

    const double k0 = 1.1;
    const double k1 = 0.4;

    double ratio = (double)area_bound.width / area_bound.height;

    return ratio > k0 || ratio < k1 ||
           (double)area.size() / threshold_image.total() < 0.006 ||
           area_bound.height == threshold_image.size().height;
  }), areas.end());


  threshold_image = cv::Mat(threshold_image.size(), threshold_image.type());
  for (auto &area: areas)
    draw_area(threshold_image, area, 255);

  cv::imshow("threshold_image", threshold_image);

  tesseract::TessBaseAPI tess_api;
  tess_api.Init("tessdata", "eng");
  tess_api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  tess_api.SetVariable("tessedit_char_whitelist", "ABCEHKMOPTXY1234567890");

  tess_api.SetImage(threshold_image.ptr(),
                    threshold_image.size().width,
                    threshold_image.size().height,
                    threshold_image.elemSize(),
                    threshold_image.step1());
  char *text = tess_api.GetUTF8Text();
  this->number_plate_text_ = text;
  delete[] text;

  this->number_plate_text_.erase(std::remove_if(this->number_plate_text_.begin(),
                                                this->number_plate_text_.end(),
                                                isspace),
                                 this->number_plate_text_.end());


  boost::match_results<std::string::iterator> what;
  if (boost::regex_search(this->number_plate_text_.begin(),
                          this->number_plate_text_.end(),
                          what,
                          boost::regex("[ABCEHKMOPTXY0][[:digit:]]{3}[ABCEHKMOPTXY0]{2}[[:digit:]]{2,3}")))
  {
    int zero_test_indexes[] = {0, 4, 5};
    for (auto index: zero_test_indexes)
      if (*(what[0].first + index) == '0')
        *(what[0].first + index) = 'O';

    this->number_plate_text_ = std::string(what[0].first, what[0].second);
  }
}
