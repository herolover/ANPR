#include "help_opencv.h"

#include "help_alg.h"

#include <queue>


double vec_RMS(const cv::Mat &vec)
{
  double square_sum = 0.0;
  for (unsigned i = 0; i < vec.total(); ++i)
    square_sum += sqr(vec.at<unsigned char>(i) / 255.0);

  return sqrt(square_sum / vec.cols);
}

double vec_sum(const cv::Mat &vec)
{
  double sum = 0.0;
  for (unsigned i = 0; i < vec.total(); ++i)
    sum += vec.at<unsigned char>(i) / 255.0;

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

void draw_area(cv::Mat &dst_img, std::vector<cv::Point> &area, unsigned char color)
{
  for (auto &point: area)
    dst_img.at<unsigned char>(point) = color;
}

void find_filled_area(cv::Mat &threshold_img, std::vector<cv::Point> &area,
                  const cv::Point &start_point, unsigned char pixel_value)
{
  std::queue<cv::Point> points;
  points.push(start_point);

  while (!points.empty())
  {
    const cv::Point &p = points.front();

    if (threshold_img.at<unsigned char>(p) == pixel_value)
    {
      threshold_img.at<unsigned char>(p) = ~pixel_value;
      area.push_back(p);

      if (p.x - 1 >= 0 &&
          threshold_img.at<unsigned char>(p + cv::Point(-1,  0)) == pixel_value)
      {
        points.push(p + cv::Point(-1,  0));
      }
      if (p.x + 1 < threshold_img.cols &&
          threshold_img.at<unsigned char>(p + cv::Point( 1,  0)) == pixel_value)
      {
        points.push(p + cv::Point( 1,  0));
      }
      if (p.y - 1 >= 0 &&
          threshold_img.at<unsigned char>(p + cv::Point( 0, -1)) == pixel_value)
      {
        points.push(p + cv::Point( 0, -1));
      }
      if (p.y + 1 < threshold_img.rows &&
          threshold_img.at<unsigned char>(p + cv::Point( 0,  1)) == pixel_value)
      {
        points.push(p + cv::Point( 0,  1));
      }
    }

    points.pop();
  }
}

std::vector<std::vector<cv::Point> > find_filled_areas(cv::Mat threshold_img,
                                                       unsigned char pixel_value)
{
  std::vector<std::vector<cv::Point> > areas;

  for (int x = 0; x < threshold_img.cols; ++x)
    for (int y = 0; y < threshold_img.rows; ++y)
      if (threshold_img.at<unsigned char>(y, x) == pixel_value)
      {
        std::vector<cv::Point> area;
        find_filled_area(threshold_img, area, cv::Point(x, y), pixel_value);
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

double compute_skew_correction_angle(const cv::Mat &image, int threshold)
{
  cv::Mat horizontal_edge_image = compute_edge_image(image, ET_HORIZONTAL);

  std::vector<cv::Vec2f> lines;
  cv::HoughLines(horizontal_edge_image, lines, 1.0, CV_PI / 360.0, threshold);

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

cv::Mat compute_edge_image(const cv::Mat &image, EdgeType edge_type)
{
  double m[3][5] = {{-0.5, -1.0, 0.0, 1.0, 0.5},
                    {-1.0, -2.0, 0.0, 2.0, 1.0},
                    {-0.5, -1.0, 0.0, 1.0, 0.5}};
  cv::Mat edge_matrix(3, 5, CV_64FC1, m);

  if (edge_type == ET_HORIZONTAL)
    edge_matrix = edge_matrix.t();

  cv::Mat edge_image_plus;
  cv::filter2D(image, edge_image_plus, -1, edge_matrix);

  cv::Mat edge_image_minus;
  cv::filter2D(image, edge_image_minus, -1, -edge_matrix);

  cv::Mat edge_image = edge_image_plus + edge_image_minus;

  cv::Mat threshold_edge_image;
  adaptive_threshold(edge_image, threshold_edge_image, 200);

  return threshold_edge_image;
}

cv::Mat make_skew_matrix(double angle, double skew_center)
{
  cv::Mat skew_matrix = cv::Mat::eye(2, 3, CV_64FC1);
  skew_matrix.at<double>(1, 0) = tan(CV_PI * 0.5 - angle);
  skew_matrix.at<double>(1, 2) = -skew_center / tan(angle);

  return skew_matrix;
}

double is_rectangle(const std::vector<cv::Point> &area)
{
  double coeffs[][10] = {
    {1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0},
    {1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0},
    {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0},
    {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0},
    {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0},
    {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0},
    {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0},
    {1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0},
    {1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0},
    {1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0}
  };

  int counts[10][10];
  for (int i = 0; i < 10; ++i)
  {
    for (int j = 0; j < 10; ++j)
    {
      counts[i][j] = 0;
    }
  }

  cv::Rect area_bound = cv::boundingRect(area);
  double w = area_bound.width / 10.0;
  double h = area_bound.height / 10.0;

  for (auto &point: area)
  {
    int x = (point.x - area_bound.x) / w;
    int y = (point.y - area_bound.y) / h;
    counts[y][x] += 1;
  }

  double res = 0.0;
  double max_res = 38.0;
  for (int i = 0; i < 10; ++i)
  {
    for (int j = 0; j < 10; ++j)
    {
      res += counts[i][j] / w / h * coeffs[i][j];
    }
  }

  return res / max_res;
}
