#include "ANPR.h"

#include <string>

#include <boost/format.hpp>
#include <tesseract/baseapi.h>

#include "help_alg.h"
#include "help_opencv.h"


#include <stdio.h>


void ANPR::set_image(const cv::Mat &image)
{
  this->image_ = image.clone();
}


void ANPR::run()
{
  auto rects = this->find_contrast_rects(this->image_, 10);
  cv::Mat skew_matrix = this->compute_skew_correction_matrix(this->image_(rects[0]));

  cv::Mat deskewed_image;
  cv::warpAffine(this->image_, deskewed_image, skew_matrix, this->image_.size());

  auto correct_rects = this->find_contrast_rects(deskewed_image, 3);

  if (correct_rects.size() > 0)
  {
    this->number_plate_image_ = deskewed_image(correct_rects[0]);
    this->number_plate_text_ = this->recognize_text(this->number_plate_image_);
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


cv::Mat ANPR::compute_edge_image(const cv::Mat &image, ANPR::EdgeType edge_type) const
{
  double m[3][3] = {{-1.0, 0.0, 1.0},
                    {-2.0, 0.0, 2.0},
                    {-1.0, 0.0, 1.0}};
  cv::Mat edge_matrix(3, 3, CV_64FC1, m);

  if (edge_type == ET_HORIZONTAL)
    edge_matrix = edge_matrix.t();

  cv::Mat edge_image;
  cv::filter2D(image, edge_image, -1, edge_matrix);

  cv::Mat threshold_edge_image;
  adaptive_threshold(edge_image, threshold_edge_image, 150);

  return threshold_edge_image;
}


cv::Mat ANPR::compute_skew_correction_matrix(const cv::Mat &image) const
{
  cv::Mat proc_mage = this->preprocess_image(image);
  cv::Mat horizontal_edge_image = this->compute_edge_image(proc_mage,
                                                           ET_HORIZONTAL);

  std::vector<cv::Vec2f> lines;
  cv::HoughLines(horizontal_edge_image, lines, 1.0, CV_PI / 360.0, 230);

  cv::Mat skew_matrix = cv::Mat::eye(2, 3, CV_64FC1);;
  if (lines.size() > 0)
  {
    double square_sum = 0.0;
    for (auto &line: lines)
      square_sum += sqr(line[1]);
    double rms_angle = sqrt(square_sum / lines.size());
    skew_matrix.at<double>(1, 0) = tan(CV_PI * 0.5 - rms_angle);
  }

  return skew_matrix;
}


cv::Mat ANPR::preprocess_image(const cv::Mat &image) const
{
  cv::Mat grayscale_image;
  cv::cvtColor(image, grayscale_image, CV_RGB2GRAY);

  cv::Mat denoised_image;
  // quite slow function
//  cv::fastNlMeansDenoising(grayscale_img, denoised_img);
  denoised_image = grayscale_image;

  return grayscale_image;
}


std::vector<cv::Rect> ANPR::find_contrast_rects(const cv::Mat &image,
                                                int margin) const
{
  int width = 640;
  double ratio = (double)image.cols / width;
  cv::Size2f size(width, image.rows / ratio);

  cv::Mat small_image;
  cv::resize(image, small_image, size);

  cv::Mat proc_image = this->preprocess_image(small_image);
  cv::Mat vertical_edge_image = this->compute_edge_image(proc_image, ET_VERTICAL);

  std::vector<double> rows_rms_contrast;
  for (int i = 0; i < vertical_edge_image.rows; ++i)
    rows_rms_contrast.push_back(row_RMS(vertical_edge_image.row(i)));

  rows_rms_contrast = simple_moving_average(rows_rms_contrast, 10, 0.0);

  save_to_file("rows", rows_rms_contrast.begin(), rows_rms_contrast.end());

  const int img_top_offset = proc_image.rows * 1 / 4;
  const int img_bottom_offset = proc_image.rows * 1 / 9;
  auto rows_rms_contrast_pairs = find_local_pairs(rows_rms_contrast.begin() + img_top_offset,
                                                  rows_rms_contrast.end() - img_bottom_offset,
                                                  0.4);

  std::vector<cv::Rect> rects;
  for (unsigned i = 0; i < rows_rms_contrast_pairs.size(); ++i)
  {
    int top_bound = std::distance(rows_rms_contrast.begin(),
                                  rows_rms_contrast_pairs[i].first) - margin;
    int bottom_bound = std::distance(rows_rms_contrast.begin(),
                                     rows_rms_contrast_pairs[i].second) + margin;

    cv::Mat both_edge_image = vertical_edge_image +
                              this->compute_edge_image(proc_image, ET_HORIZONTAL);


    cv::Mat rows_range = both_edge_image.rowRange(top_bound, bottom_bound);

    std::vector<double> cols_rms_contrast;
    for (int j = 0; j < rows_range.cols; ++j)
      cols_rms_contrast.push_back(col_RMS(rows_range.col(j)));

    cols_rms_contrast = simple_moving_average(cols_rms_contrast, 30, 0.0);

    save_to_file(boost::str(boost::format("cols_%1%") % i),
                 cols_rms_contrast.begin(), cols_rms_contrast.end());

    const int img_offset_left = proc_image.cols / 8;
    const int img_offset_right = proc_image.cols / 8;
    auto cols_rms_contrast_pairs = find_local_pairs(cols_rms_contrast.begin() + img_offset_left,
                                                    cols_rms_contrast.end() - img_offset_right,
                                                    0.05);

    for (auto &cols_pair: cols_rms_contrast_pairs)
    {
      int left_bound = std::distance(cols_rms_contrast.begin(), cols_pair.first) - margin;
      int right_bound = std::distance(cols_rms_contrast.begin(), cols_pair.second) + margin;

      cv::Rect rect;

      rect.x = left_bound;
      rect.y = top_bound;
      rect.height = bottom_bound - top_bound;
      rect.width = right_bound - left_bound;

      rects.push_back(rect * ratio);
    }
  }

  return rects;
}


std::string ANPR::recognize_text(const cv::Mat &image) const
{
  cv::Mat proc_image = this->preprocess_image(image);

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

    const double k0 = 1.2;
    const double k1 = 0.4;

    double ratio = (double)area_bound.width / area_bound.height;

    return ratio > k0 || ratio < k1 ||
           (double)area.size() / threshold_image.total() < 0.006 ||
           area_bound.height == threshold_image.size().height;
  }), areas.end());


  threshold_image = cv::Mat(threshold_image.size(), threshold_image.type());
  for (auto &area: areas)
    draw_area(threshold_image, area, 255);

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
  plate_number = text;
  delete[] text;

  cv::imshow("threshold image", threshold_image);
  cv::imwrite("plate.png", threshold_image);

  return plate_number;
}
