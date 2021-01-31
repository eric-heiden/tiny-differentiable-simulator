// Copyright (c) 2017 GeometryFactory Sarl (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL: https://github.com/CGAL/cgal/blob/v5.2/Classification/include/CGAL/Classification/ETHZ/Random_forest_classifier.h $
// $Id: Random_forest_classifier.h 19004a7 2020-08-04T13:41:48+02:00 Simon Giraudot
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s)     : Simon Giraudot

#ifndef CGAL_CLASSIFICATION_ETHZ_RANDOM_FOREST_CLASSIFIER_H
#define CGAL_CLASSIFICATION_ETHZ_RANDOM_FOREST_CLASSIFIER_H

#include <CGAL/license/Classification.h>

#include <CGAL/Classification/Feature_set.h>
#include <CGAL/Classification/Label_set.h>
#include <CGAL/Classification/internal/verbosity.h>

#ifdef CGAL_CLASSIFICATION_VERBOSE
#define VERBOSE_TREE_PROGRESS 1
#endif

// Disable warnings from auxiliary library
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable:4141)
#  pragma warning(disable:4244)
#  pragma warning(disable:4267)
#  pragma warning(disable:4275)
#  pragma warning(disable:4251)
#  pragma warning(disable:4996)
#endif

#include <CGAL/Classification/ETHZ/internal/random-forest/node-gini.hpp>
#include <CGAL/Classification/ETHZ/internal/random-forest/forest.hpp>

#include <CGAL/tags.h>

#if defined(CGAL_LINKED_WITH_BOOST_IOSTREAMS) && defined(CGAL_LINKED_WITH_BOOST_SERIALIZATION)
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#endif

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif


namespace CGAL {

namespace Classification {

namespace ETHZ {

/*!
  \ingroup PkgClassificationClassifiersETHZ

  \brief %Classifier based on the ETH Zurich version of the random forest algorithm \cgalCite{cgal:w-erftl-14}.

  \note This classifier is distributed under the MIT license.

  \cgalModels `CGAL::Classification::Classifier`
*/
class Random_forest_classifier
{
  typedef CGAL::internal::liblearning::RandomForest::RandomForest
  < CGAL::internal::liblearning::RandomForest::NodeGini
    < CGAL::internal::liblearning::RandomForest::AxisAlignedSplitter> > Forest;

  const Label_set& m_labels;
  const Feature_set& m_features;
  std::shared_ptr<Forest> m_rfc;

public:

  /// \name Constructor
  /// @{

  /*!
    \brief instantiates the classifier using the sets of `labels` and `features`.

  */
  Random_forest_classifier (const Label_set& labels,
                            const Feature_set& features)
    : m_labels (labels), m_features (features)
  { }

  /*!
    \brief copies the `other` classifier's configuration using another
    set of `features`.

    This constructor can be used to apply a trained random forest to
    another data set.

    \warning The feature set should be composed of the same features
    than the ones used by `other`, and in the same order.

  */
#if defined(DOXYGEN_RUNNING) || \
  (defined(CGAL_LINKED_WITH_BOOST_IOSTREAMS) && \
   defined(CGAL_LINKED_WITH_BOOST_SERIALIZATION))
  Random_forest_classifier (const Random_forest_classifier& other,
                            const Feature_set& features)
    : m_labels (other.m_labels), m_features (features)
  {
    std::stringstream stream;
    other.save_configuration(stream);
    this->load_configuration(stream);
  }
#endif

  /// @}

  /// \name Training
  /// @{

  /// \cond SKIP_IN_MANUAL
  template <typename LabelIndexRange>
  void train (const LabelIndexRange& ground_truth,
              bool reset_trees = true,
              std::size_t num_trees = 25,
              std::size_t max_depth = 20)
  {
    train<CGAL::Parallel_if_available_tag>(ground_truth, reset_trees, num_trees, max_depth);
  }
  /// \endcond

  /*!
    \brief runs the training algorithm.

    From the set of provided ground truth, this algorithm estimates
    sets up the random trees that produce the most accurate result
    with respect to this ground truth.

    \pre At least one ground truth item should be assigned to each
    label.

    \tparam ConcurrencyTag enables sequential versus parallel
    algorithm. Possible values are `Parallel_tag` (default value if
    %CGAL is linked with TBB) or `Sequential_tag` (default value
    otherwise).

    \param ground_truth vector of label indices. It should contain for
    each input item, in the same order as the input set, the index of
    the corresponding label in the `Label_set` provided in the
    constructor. Input items that do not have a ground truth
    information should be given the value `-1`.

    \param reset_trees should be set to `false` if the users wants to
    _add_ new trees to the existing forest, and kept to `true` if the
    training should be recomputing from scratch (discarding the
    current forest).

    \param num_trees number of trees generated by the training
    algorithm. Higher values may improve result at the cost of higher
    computation times (in general, using a few dozens of trees is
    enough).

    \param max_depth maximum depth of the trees. Higher values will
    improve how the forest fits the training set. A overly low value
    will underfit the test data and conversely an overly high value
    will likely overfit.
  */
  template <typename ConcurrencyTag, typename LabelIndexRange>
  void train (const LabelIndexRange& ground_truth,
              bool reset_trees = true,
              std::size_t num_trees = 25,
              std::size_t max_depth = 20)
  {
    CGAL_precondition (m_labels.is_valid_ground_truth (ground_truth));

    CGAL::internal::liblearning::RandomForest::ForestParams params;
    params.n_trees   = num_trees;
    params.max_depth = max_depth;

    std::vector<int> gt;
    std::vector<float> ft;

#ifdef CGAL_CLASSIFICATION_VERBOSE
    std::vector<std::size_t> count (m_labels.size(), 0);
#endif

    std::size_t i = 0;
    for (const auto& gt_value : ground_truth)
    {
      int g = int(gt_value);
      if (g != -1)
      {
        for (std::size_t f = 0; f < m_features.size(); ++ f)
          ft.push_back(m_features[f]->value(i));
        gt.push_back(g);
#ifdef CGAL_CLASSIFICATION_VERBOSE
        count[std::size_t(g)] ++;
#endif
      }
      ++ i;
    }

    CGAL_CLASSIFICATION_CERR << "Using " << gt.size() << " inliers:" << std::endl;
#ifdef CGAL_CLASSIFICATION_VERBOSE
    for (std::size_t i = 0; i < m_labels.size(); ++ i)
      std::cerr << " * " << m_labels[i]->name() << ": " << count[i] << " inlier(s)" << std::endl;
#endif

    CGAL::internal::liblearning::DataView2D<int> label_vector (&(gt[0]), gt.size(), 1);
    CGAL::internal::liblearning::DataView2D<float> feature_vector(&(ft[0]), gt.size(), ft.size() / gt.size());

    if (m_rfc && reset_trees)
      m_rfc.reset();

    if (!m_rfc)
      m_rfc = std::make_shared<Forest> (params);

    CGAL::internal::liblearning::RandomForest::AxisAlignedRandomSplitGenerator generator;

    m_rfc->train<ConcurrencyTag>
      (feature_vector, label_vector, CGAL::internal::liblearning::DataView2D<int>(), generator, 0, reset_trees, m_labels.size());
  }

  /// \cond SKIP_IN_MANUAL
  void operator() (std::size_t item_index, std::vector<float>& out) const
  {
    out.resize (m_labels.size(), 0.);

    std::vector<float> ft;
    ft.reserve (m_features.size());
    for (std::size_t f = 0; f < m_features.size(); ++ f)
      ft.push_back (m_features[f]->value(item_index));

    std::vector<float> prob (m_labels.size());

    m_rfc->evaluate (ft.data(), prob.data());

    for (std::size_t i = 0; i < out.size(); ++ i)
      out[i] = (std::min) (1.f, (std::max) (0.f, prob[i]));
  }

  /// \endcond

  /// @}

  /// \name Miscellaneous
  /// @{

  /*!
    \brief computes, for each feature, how many nodes in the forest
    uses it as a split criterion.

    Each tree of the random forest recursively splits the training
    data set using at each node one of the input features. This method
    counts, for each feature, how many times it was selected by the
    training algorithm as a split criterion.

    This method allows to evaluate how useful a feature was with
    respect to a training set: if a feature is used a lot, that means
    that it has a strong discriminative power with respect to how the
    labels are represented by the feature set; on the contrary, if a
    feature is not used very often, its discriminative power is
    probably low; if a feature is _never_ used, it likely has no
    interest at all and is completely uncorrelated to the label
    segmentation of the training set.

    \param count vector where the result is stored. After running the
    method, it contains, for each feature, the number of nodes in the
    forest that use it as a split criterion, in the same order as the
    feature set order.
  */
  void get_feature_usage (std::vector<std::size_t>& count) const
  {
    count.clear();
    count.resize(m_features.size(), 0);
    return m_rfc->get_feature_usage(count);
  }

  /// @}

  /// \name Input/Output
  /// @{

  /*!
    \brief saves the current configuration in the stream `output`.

    This allows to easily save and recover a specific classification
    configuration.

    The output file is written in a binary format that is readable by
    the `load_configuration()` method.
  */
#if defined(DOXYGEN_RUNNING) || \
  (defined(CGAL_LINKED_WITH_BOOST_IOSTREAMS) && \
   defined(CGAL_LINKED_WITH_BOOST_SERIALIZATION))
  void save_configuration (std::ostream& output) const
  {
    m_rfc->write(output);
  }
#endif

  /*!
    \brief loads a configuration from the stream `input`.

    The input file should be a binary file written by the
    `save_configuration()` method. The feature set of the classifier
    should contain the exact same features in the exact same order as
    the ones present when the file was generated using
    `save_configuration()`.

    \warning If the file you are trying to load was saved using CGAL
    5.1 or earlier, you have to convert it first using
    `convert_deprecated_configuration_to_new_format()` as the exchange
    format for ETHZ Random Forest changed in CGAL 5.2.

  */
#if defined(DOXYGEN_RUNNING) || \
  (defined(CGAL_LINKED_WITH_BOOST_IOSTREAMS) && \
   defined(CGAL_LINKED_WITH_BOOST_SERIALIZATION))
  void load_configuration (std::istream& input)
  {
    CGAL::internal::liblearning::RandomForest::ForestParams params;
    m_rfc = std::make_shared<Forest> (params);

    m_rfc->read(input);
  }
#endif

  /// @}

  /// \name Deprecated Input/Output
  /// @{

  /*!
    \brief converts a deprecated configuration (in compressed ASCII
    format) to a new configuration (in binary format).

    The input file should be a GZIP container written by the
    `save_configuration()` method from CGAL 5.1 and earlier. The
    output is a valid configuration for CGAL 5.2 and later.

    \note This function depends on the Boost libraries
    [Serialization](https://www.boost.org/libs/serialization) and
    [IO Streams](https://www.boost.org/libs/iostreams) (compiled with the GZIP dependency).
  */
#if defined(DOXYGEN_RUNNING) || \
  (defined(CGAL_LINKED_WITH_BOOST_IOSTREAMS) && \
   defined(CGAL_LINKED_WITH_BOOST_SERIALIZATION))
  static void convert_deprecated_configuration_to_new_format (std::istream& input, std::ostream& output)
  {
    Label_set dummy_labels;
    Feature_set dummy_features;
    Random_forest_classifier classifier (dummy_labels, dummy_features);
    classifier.load_deprecated_configuration(input);
    classifier.save_configuration(output);
  }
#endif

/// @}

  /// \cond SKIP_IN_MANUAL
#if defined(CGAL_LINKED_WITH_BOOST_IOSTREAMS) && \
  defined(CGAL_LINKED_WITH_BOOST_SERIALIZATION)
  void load_deprecated_configuration (std::istream& input)
  {
    CGAL::internal::liblearning::RandomForest::ForestParams params;
    m_rfc = std::make_shared<Forest> (params);

    boost::iostreams::filtering_istream ins;
    ins.push(boost::iostreams::gzip_decompressor());
    ins.push(input);
    boost::archive::text_iarchive ias(ins);
    ias >> BOOST_SERIALIZATION_NVP(*m_rfc);
  }
#endif
  /// \endcond


};

}

/// \cond SKIP_IN_MANUAL
// Backward compatibility
typedef ETHZ::Random_forest_classifier ETHZ_random_forest_classifier;
/// \endcond

}

}

#endif // CGAL_CLASSIFICATION_ETHZ_RANDOM_FOREST_CLASSIFIER_H
