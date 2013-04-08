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

  auto rect = this->find_number_plate_rect(deskewed_image);
  this->number_plate_image_ = deskewed_image(rect);
  this->recognize_text();
}


cv::Mat ANPR::get_number_plate_image() const
{
  return this->number_plate_image_;
}


std::string ANPR::get_number_plate_text() const
{
  return this->number_plate_text_;
}


cv::Rect ANPR::find_number_plate_rect(const cv::Mat &image) const
{
  cv::Mat cropped_image = image(this->search_rect_);

  cv::Mat proc_image = convert_to_grayscale_and_remove_noise(cropped_image);
  cv::Mat vertical_edge_image = compute_edge_image(proc_image, ET_VERTICAL);

//  cv::imshow("vertical_edge_image", vertical_edge_image);

  std::vector<double> rows_rms_contrast;
  for (int i = 0; i < vertical_edge_image.rows; ++i)
    rows_rms_contrast.push_back(vec_RMS(vertical_edge_image.row(i)));

  rows_rms_contrast = median_smooth(rows_rms_contrast, 5, 0.0);

//  save_to_file("rows", rows_rms_contrast.begin(), rows_rms_contrast.end());


  auto rows_rms_contrast_pairs = find_local_pairs(rows_rms_contrast.begin(),
                                                  rows_rms_contrast.end(),
                                                  0.3);
  auto wided_pair = *std::max_element(rows_rms_contrast_pairs.begin(),
                                      rows_rms_contrast_pairs.end(),
                                      [](const std::pair<std::vector<double>::iterator,
                                                         std::vector<double>::iterator> &a,
                                         const std::pair<std::vector<double>::iterator,
                                                         std::vector<double>::iterator> &b)
  {
    return std::distance(a.first, a.second) < std::distance(b.first, b.second);
  });



  int top_bound = std::distance(rows_rms_contrast.begin(),
                                wided_pair.first);
  int bottom_bound = std::distance(rows_rms_contrast.begin(),
                                   wided_pair.second);

  cv::Mat both_edge_image = vertical_edge_image +
                            compute_edge_image(proc_image, ET_HORIZONTAL);

//  cv::imshow("both_edge_image", both_edge_image);

  cv::Mat rows_range = both_edge_image.rowRange(top_bound, bottom_bound);

  std::vector<double> cols_rms_contrast;
  for (int j = 0; j < rows_range.cols; ++j)
    cols_rms_contrast.push_back(vec_RMS(rows_range.col(j)));

//  save_to_file("cols", cols_rms_contrast.begin(), cols_rms_contrast.end());

  cols_rms_contrast = simple_moving_average(cols_rms_contrast, 60, 0.0);

//  save_to_file("cols_s", cols_rms_contrast.begin(), cols_rms_contrast.end());

  auto cols_rms_contrast_pairs = find_local_pairs(cols_rms_contrast.begin(),
                                                  cols_rms_contrast.end(),
                                                  0.03);

  std::sort(cols_rms_contrast_pairs.begin(), cols_rms_contrast_pairs.end(),
            [](const std::pair<std::vector<double>::iterator,
                               std::vector<double>::iterator> &a,
               const std::pair<std::vector<double>::iterator,
                               std::vector<double>::iterator> &b)
  {
    return std::distance(a.first, a.second) > std::distance(b.first, b.second);
  });

  int left_bound = std::distance(cols_rms_contrast.begin(),
                                 cols_rms_contrast_pairs.front().first);
  int right_bound = std::distance(cols_rms_contrast.begin(),
                                  cols_rms_contrast_pairs.front().second);

  cv::Rect rect;
  rect.x = left_bound + this->search_rect_.x;
  rect.y = top_bound + this->search_rect_.y;
  rect.height = bottom_bound - top_bound;
  rect.width = right_bound - left_bound;

  return rect;
}


void ANPR::recognize_text()
{
  cv::Mat proc_image = convert_to_grayscale_and_remove_noise(this->number_plate_image_);

//  cv::imshow("proc_image", proc_image);

  cv::Mat blured_image;
  cv::GaussianBlur(proc_image, blured_image, cv::Size(101, 101), 0);

  cv::Mat equalized_image;
  cv::divide(proc_image, blured_image, equalized_image, 256.0);

//  cv::imshow("equalized_image", equalized_image);

  cv::Mat threshold_image;
  cv::threshold(equalized_image, threshold_image, 210.0, 255.0, CV_THRESH_BINARY_INV);

  // fill a small noise on a characters
  std::vector<std::vector<cv::Point> > noisy_areas = find_filled_areas(threshold_image.clone(), 0);
  for (auto &area: noisy_areas)
    if (area.size() < 60)
      draw_area(threshold_image, area, 255);

//  cv::imshow("threshold_image_prev", threshold_image);


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

//  cv::imshow("threshold_image", threshold_image);

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
