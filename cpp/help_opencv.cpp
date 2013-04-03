#include "help_opencv.h"

#include "help_alg.h"


double row_RMS(const cv::Mat &row)
{
  double square_sum = 0.0;
  for (int i = 0; i < row.cols; ++i)
  {
    double value = row.at<unsigned char>(0, i) / 255.0;
    square_sum += (double)value * value;
  }

  return sqrt(square_sum / row.cols);
}

double col_RMS(const cv::Mat &col)
{
  double square_sum = 0.0;
  for (int i = 0; i < col.rows; ++i)
  {
    double value = col.at<unsigned char>(i, 0) / 255.0;
    square_sum += (double)value * value;
  }

  return sqrt(square_sum / col.rows);
}

double row_sum(const cv::Mat &row)
{
  double sum = 0.0;
  for (int i = 0; i < row.cols; ++i)
  {
    double value = row.at<unsigned char>(0, i) / 255.0;
    sum += value;
  }

  return sum;
}


double col_sum(const cv::Mat &col)
{
  double sum = 0.0;
  for (int i = 0; i < col.rows; ++i)
  {
    double value = col.at<unsigned char>(i, 0) / 255.0;
    sum += value;
  }

  return sum;
}


cv::Rect operator * (const cv::Rect &rect, double k)
{
  cv::Rect new_rect(rect);
  new_rect.x *= k;
  new_rect.y *= k;
  new_rect.width *= k;
  new_rect.height *= k;

  return new_rect;
}


void draw_area(cv::Mat &dst_img, std::vector<cv::Point> &area, int color)
{
  for (auto &point: area)
    dst_img.at<unsigned char>(point) = color;
}


void walk_on_area(cv::Mat &threshold_img, std::vector<cv::Point> &area,
                  const cv::Point &point, int pixel_value)
{
  if (point.x >= 0 && point.x < threshold_img.cols &&
      point.y >= 0 && point.y < threshold_img.rows &&
      threshold_img.at<unsigned char>(point) == pixel_value)
  {
    threshold_img.at<unsigned char>(point) = pixel_value ^ 0xff;
    area.push_back(point);

    walk_on_area(threshold_img, area, point + cv::Point(-1,  0), pixel_value);
    walk_on_area(threshold_img, area, point + cv::Point( 1,  0), pixel_value);
    walk_on_area(threshold_img, area, point + cv::Point( 0, -1), pixel_value);
    walk_on_area(threshold_img, area, point + cv::Point( 0,  1), pixel_value);
  }
}


std::vector<std::vector<cv::Point> > find_filled_areas(cv::Mat threshold_img,
                                                       int pixel_value)
{
  std::vector<std::vector<cv::Point> > areas;

  for (int x = 0; x < threshold_img.cols; ++x)
    for (int y = 0; y < threshold_img.rows; ++y)
      if (threshold_img.at<unsigned char>(y, x) == pixel_value)
      {
        std::vector<cv::Point> area;
        walk_on_area(threshold_img, area, cv::Point(x, y), pixel_value);
        areas.push_back(area);
      }

  return areas;
}


void adaptive_threshold(const cv::Mat &src_img, cv::Mat &dst_img, double thresh)
{
  src_img.copyTo(dst_img);
  for (int i = 1; i < dst_img.cols - 1; ++i)
    for (int j = 1; j < dst_img.rows - 1; ++j)
    {
      double sum = 0.0;

      sum += dst_img.at<unsigned char>(j - 1, i - 1);
      sum += dst_img.at<unsigned char>(j - 1, i - 0);
      sum += dst_img.at<unsigned char>(j - 1, i + 1);
      sum += dst_img.at<unsigned char>(j - 0, i + 1);
      sum += dst_img.at<unsigned char>(j - 0, i - 0) * 8.0;
      sum += dst_img.at<unsigned char>(j - 0, i - 1);
      sum += dst_img.at<unsigned char>(j + 1, i - 1);
      sum += dst_img.at<unsigned char>(j + 1, i - 0);
      sum += dst_img.at<unsigned char>(j + 1, i + 1);

      if (sum / 9.0 < thresh)
        dst_img.at<unsigned char>(j, i) = 0;
    }
}


double compute_skew_correction_angle(const cv::Mat &image)
{
  cv::Mat proc_mage = convert_to_grayscale_and_remove_noise(image);
  cv::Mat horizontal_edge_image = compute_edge_image(proc_mage, ET_HORIZONTAL);

  std::vector<cv::Vec2f> lines;
  cv::HoughLines(horizontal_edge_image, lines, 1.0, CV_PI / 360.0, 230);

  double skew_correction_angle = CV_PI * 0.5;
  if (lines.size() > 0)
  {
    double square_sum = 0.0;
    for (auto &line: lines)
      square_sum += sqr(line[1]);
    skew_correction_angle = sqrt(square_sum / lines.size());
  }

  return skew_correction_angle;
}


cv::Mat convert_to_grayscale_and_remove_noise(const cv::Mat &image)
{
  cv::Mat grayscale_image;
  cv::cvtColor(image, grayscale_image, CV_RGB2GRAY);

  cv::Mat denoised_image;
  // quite slow function
//  cv::fastNlMeansDenoising(grayscale_image, denoised_image);
  denoised_image = grayscale_image;

  return grayscale_image;
}


cv::Mat compute_edge_image(const cv::Mat &image, EdgeType edge_type)
{
  double m[3][3] = {{-1.0, 0.0, 1.0},
                    {-3.0, 0.0, 3.0},
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
