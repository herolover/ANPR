#include "ANPR.h"

#include <string>
#include <algorithm>
#include <unordered_map>

#include <boost/format.hpp>
#include <boost/regex.hpp>

#include <tesseract/baseapi.h>

#include "help_alg.h"
#include "help_opencv.h"

struct AreaGroup
{
  cv::Rect bound;
  std::vector<std::pair<std::vector<cv::Point>, cv::Rect> *> areas;
};

std::string ANPR::recognize_number_plate(const cv::Mat &image,
                                         const cv::Rect &search_rect)
{
  cv::Mat number_plate_image = image(search_rect);

  cv::Mat grayscale_image;
  cv::cvtColor(number_plate_image, grayscale_image, CV_RGB2GRAY);

  cv::Mat denoised_image;
  cv::medianBlur(grayscale_image, denoised_image, 3);

  cv::Mat blured_image;
  cv::GaussianBlur(denoised_image, blured_image, cv::Size(31, 31), 0);
//  cv::imshow("blured_image", blured_image);

  cv::Mat grained_image = denoised_image - blured_image + cv::Scalar(128);
//  cv::imshow("grained_image", grained_image);

  cv::Mat threshold_image;
  cv::threshold(grained_image, threshold_image, 127.0, 255.0, CV_THRESH_BINARY_INV);

  cv::Mat group_bounds_image = cv::Mat::zeros(threshold_image.size(), CV_8UC1);

  auto areas = find_filled_areas(threshold_image, 255);
  areas.erase(std::remove_if(areas.begin(), areas.end(),
                             [&threshold_image](const std::pair<std::vector<cv::Point>, cv::Rect> &area)
  {
    double k = (double)area.second.width / area.second.height;

    return area.second.x == 0 || area.second.x + area.second.width == threshold_image.cols ||
           area.second.y == 0 || area.second.y + area.second.height == threshold_image.rows ||
           area.first.size() < 50 || k > 1.0 || k < 0.3;
  }), areas.end());

  for (auto &area: areas)
  {
    cv::Rect area_bound = area.second;
    area_bound.width += 20;
    area_bound.height += 20;
    area_bound.x -= 10;
    area_bound.y -= 10;

    cv::rectangle(group_bounds_image, area_bound, cv::Scalar(255), CV_FILLED);
  }

  auto group_bounds = find_filled_areas(group_bounds_image, 255);

  std::vector<AreaGroup> groups;
  for (auto &group_bound: group_bounds)
  {
    AreaGroup group;
    group.bound = group_bound.second;
    for (auto &area: areas)
    {
      if (is_intersects(group_bound.second, area.second))
      {
        group.areas.push_back(&area);
      }
    }
    groups.push_back(group);
  }

  groups.erase(std::remove_if(groups.begin(), groups.end(),
                              [](const AreaGroup &group)
  {
    return group.areas.size() < 8;
  }), groups.end());

  if (groups.size() == 0)
    return "";

  threshold_image = cv::Mat::zeros(threshold_image.size(), CV_8UC1);
  for (auto &group: groups)
  {
    for (auto &area: group.areas)
    {
      draw_area(threshold_image, area->first, 255);
    }
    cv::rectangle(threshold_image, group.bound, cv::Scalar(255));
  }
  cv::imshow("threshold_image", threshold_image);

  std::sort(groups.begin(), groups.end(),
            [](const AreaGroup &a, const AreaGroup &b)
  {
    auto estimate = [](const AreaGroup &group)
    {
      double res = 0.0;
      for (auto &area: group.areas)
      {
        res += sqr(group.bound.br().y - area->second.br().y);
      }

      return sqrt(res / group.areas.size());
    };

    return estimate(a) < estimate(b);
  });

  double mean_height = 0.0;
  for (auto &area: groups[0].areas)
  {
    mean_height += area->second.height;
  }
  mean_height /= groups[0].areas.size();
  std::cout << "Mean height: " << mean_height << std::endl;

  threshold_image = cv::Mat::zeros(threshold_image.size(), CV_8UC1);
  for (auto &area: groups[0].areas)
  {
    double k = area->second.height / mean_height;
    std::cout << area->second.height << " " << k << std::endl;
    if (k > 0.8)
    {
      draw_area(threshold_image, area->first, 255);
    }
  }
  cv::imshow("threshold_image2", threshold_image);

  tesseract::TessBaseAPI tess_api;
  tess_api.Init("/home/anton/projects/ANPR/ANPR-build/qtc_Desktop-release/tessdata", "rus");
  tess_api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);

  tess_api.SetImage(threshold_image.ptr(),
                    threshold_image.size().width,
                    threshold_image.size().height,
                    threshold_image.elemSize(),
                    threshold_image.step1());
  char *text = tess_api.GetUTF8Text();
  std::string number_plate_text(text);
  delete[] text;

  number_plate_text.erase(std::remove_if(number_plate_text.begin(),
                                         number_plate_text.end(),
                                         isspace),
                          number_plate_text.end());

  boost::match_results<std::string::iterator> what;
  if (boost::regex_search(number_plate_text.begin(),
                          number_plate_text.end(),
                          what,
                          boost::regex("[ABCEHKMOPTXY08][1234567890BO]{3}[ABCEHKMOPTXY08]{2}[1234567890BO]{0,3}")))
  {
    std::unordered_map<char, char> digits = {
      {'O', '0'},
      {'B', '8'}
    };
    int digits_pos[] = {1, 2, 3, 6, 7, 8};
    for (auto pos: digits_pos)
    {
      char &symbol = *(what[0].first + pos);
      if (digits.count(symbol) == 1)
      {
        symbol = digits[symbol];
      }
    }

    std::unordered_map<char, char> letters = {
      {'0', 'O'},
      {'8', 'B'}
    };
    int letters_pos[] = {0, 4, 5};
    for (auto pos: letters_pos)
    {
      if (what[0].first + pos < number_plate_text.end())
      {
        char &symbol = *(what[0].first + pos);
        if (letters.count(symbol) == 1)
        {
          symbol = letters[symbol];
        }
      }
    }

    number_plate_text = std::string(what[0].first, what[0].second);
  }

  return number_plate_text;
}
