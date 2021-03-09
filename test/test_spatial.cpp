#include <gtest/gtest.h>

// #define TDS_USE_LEFT_ASSOCIATIVE_TRANSFORMS true

#include "dynamics/forward_dynamics.hpp"
#include "math/eigen_algebra.hpp"
#include "math/tiny/tiny_algebra.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "rbdl_test_utils.hpp"

template <typename Algebra>
void test_transform_matrix_consistency() {
  using Transform = tds::Transform<Algebra>;
  using Vector3 = typename Algebra::Vector3;
  using Matrix6 = typename Algebra::Matrix6;

  Transform t1(Vector3(0.2, 0.6, -0.8),
               Algebra::rotation_zyx_matrix(-0.9, -0.1, 0.6));
  Transform t2(Vector3(1.9, 3.6, 8.2),
               Algebra::rotation_zyx_matrix(3.9, 0.63, 1.7));
  Matrix6 tf_result = (t1 * t2).matrix();
// #ifdef TDS_USE_LEFT_ASSOCIATIVE_TRANSFORMS
  // Matrix6 mat_result = t1.matrix() * t2.matrix();
// #else
  Matrix6 mat_result = t2.matrix() * t1.matrix();
// #endif

  // Algebra::print("tf_result", tf_result);
  // Algebra::print("mat_result", mat_result);

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(tf_result(i, j), mat_result(i, j), 1e-8);
    }
  }
}

template <typename Algebra>
void test_transform_rbdl() {
  using Transform = tds::Transform<Algebra>;
  using Vector3 = typename Algebra::Vector3;
  using Matrix6 = typename Algebra::Matrix6;

  // Transform t1(Vector3(0.2, 0.6, -0.8),
  //              Algebra::rotation_zyx_matrix(-0.9, -0.1, 0.6));
  // auto rbdl_t1 = to_rbdl<Algebra>(t1);
  // EXPECT_TRUE(is_equal<Algebra>(t1, rbdl_t1));
  // Transform t2(Vector3(1.9, 3.6, 8.2),
  //              Algebra::rotation_zyx_matrix(3.9, 0.63, 1.7));
  Transform t1(Vector3(1.0, 2.0, 3.0),
               Algebra::rotation_x_matrix(M_PI_2));
  auto rbdl_t1 = to_rbdl<Algebra>(t1);
  Algebra::print("\nTDS t1:", t1);
  std::cout << "RBDL T1: " << rbdl_t1 << std::endl;
  EXPECT_TRUE(is_equal<Algebra>(t1, rbdl_t1));
  auto t1_m = t1.matrix();
  auto rbdl_t1_m = rbdl_t1.toMatrix();
  Algebra::print("\nTDS t1 MATRIX:", t1_m);
  std::cout << "RBDL T1 MATRIX:\n" << rbdl_t1_m << std::endl;
  EXPECT_TRUE(is_equal<Algebra>(t1_m, rbdl_t1_m));
  Transform t2(Vector3(3.0, 2.0 ,1.0),
               Algebra::rotation_x_matrix(M_PI_2));
  auto rbdl_t2 = to_rbdl<Algebra>(t2);
  Transform t12 = t1 * t2;
  auto rbdl_t12 = rbdl_t2 * rbdl_t1;
  // auto rbdl_t12 = rbdl_t2 * rbdl_t1;
  Algebra::print("\nTDS t12:", t12);
  std::cout << "RBDL T12: " << rbdl_t12 << std::endl;
  auto t12_m = t12.matrix();
  auto rbdl_t12_m = rbdl_t12.toMatrix();
  Algebra::print("\nTDS t12 MATRIX:", t12_m);
  std::cout << "RBDL t12 MATRIX:\n" << rbdl_t12_m << std::endl;
  EXPECT_TRUE(is_equal<Algebra>(t12, rbdl_t12));
}

template <typename Algebra>
void test_transform_inverse() {
  using Transform = tds::Transform<Algebra>;
  using Vector3 = typename Algebra::Vector3;
  using Matrix6 = typename Algebra::Matrix6;

  Transform t(Vector3(0.2, 0.6, -0.8),
              Algebra::rotation_zyx_matrix(-0.9, -0.1, 0.6));

  Matrix6 tf_result1 = (t * t.inverse()).matrix();
  Matrix6 tf_result2 = (t.inverse() * t).matrix();
  Algebra::print("t * t.inverse()", tf_result1);
  Algebra::print("t.inverse() * t", tf_result2);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (i == j) {
        EXPECT_NEAR(tf_result1(i, j), 1.0, 1e-8);
        EXPECT_NEAR(tf_result2(i, j), 1.0, 1e-8);
      } else {
        EXPECT_NEAR(tf_result1(i, j), 0.0, 1e-8);
        EXPECT_NEAR(tf_result2(i, j), 0.0, 1e-8);
      }
    }
  }
}

template <typename Algebra>
void test_transform_transpose() {
  using Transform = tds::Transform<Algebra>;
  using Vector3 = typename Algebra::Vector3;
  using Matrix6 = typename Algebra::Matrix6;

  Transform t(Vector3(0.2, 0.6, -0.8),
              Algebra::rotation_zyx_matrix(-0.9, -0.1, 0.6));

  Matrix6 tf_result = t.matrix();
  Matrix6 tft_result = t.matrix_transpose();
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(tf_result(i, j), tft_result(j, i), 1e-8);
    }
  }
}

template <typename Algebra>
inline Eigen::Matrix<double, 6, 1> to_eigen(
    const tds::SpatialVector<Algebra>& sv) {
  Eigen::Matrix<double, 6, 1> v;
  for (int i = 0; i < 6; ++i) {
    v[i] = Algebra::to_double(sv[i]);
  }
  return v;
}

inline Eigen::Matrix<double, 6, 6> to_eigen(
    const Eigen::Matrix<double, 6, 6>& mt) {
  return mt;
}

inline Eigen::Matrix<double, 3, 3> to_eigen(
    const Eigen::Matrix<double, 3, 3>& mt) {
  return mt;
}

inline Eigen::Matrix<double, 3, 1> to_eigen(
    const Eigen::Matrix<double, 3, 1>& mt) {
  return mt;
}

template <typename Scalar, typename Utils>
inline Eigen::Matrix<double, 6, 6> to_eigen(
    const TINY::TinyMatrix6x6<Scalar, Utils>& mt) {
  Eigen::Matrix<double, 6, 6> m;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      m(i, j) = mt(i, j);
    }
  }
  // mt.print("input matrix");
  // std::cout << "output Eigen matrix:\n" << m << std::endl;
  return m;
}

template <typename Scalar, typename Utils>
inline Eigen::Matrix<double, 3, 3> to_eigen(
    const TINY::TinyMatrix3x3<Scalar, Utils>& mt) {
  Eigen::Matrix<double, 3, 3> m;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m(i, j) = mt(i, j);
    }
  }
  // mt.print("input matrix");
  // std::cout << "output Eigen matrix:\n" << m << std::endl;
  return m;
}

template <typename Scalar, typename Utils>
inline Eigen::Matrix<double, 3, 1> to_eigen(
    const TINY::TinyVector3<Scalar, Utils>& mt) {
  Eigen::Matrix<double, 3, 1> m;
  for (int i = 0; i < 3; ++i) {
    m(i, 0) = mt[i];
  }
  // mt.print("input matrix");
  // std::cout << "output Eigen matrix:\n" << m << std::endl;
  return m;
}

template <typename Algebra>
void test_transform_spatial_vector() {
  using Transform = tds::Transform<Algebra>;
  using MotionVector = tds::MotionVector<Algebra>;
  using ForceVector = tds::ForceVector<Algebra>;
  using Vector3 = typename Algebra::Vector3;
  using Matrix6 = typename Algebra::Matrix6;

  Transform t(Vector3(0.2, 0.6, -0.8),
              Algebra::rotation_zyx_matrix(-0.9, -0.1, 0.6));
  MotionVector mv(Vector3(0.1, -0.6, -8.7), Vector3(0.9, 3.8, -0.1));
  ForceVector fv(Vector3(0.9, 3.8, -0.1), Vector3(0.1, -0.6, -8.7));

  // Algebra::print("Transform matrix:", t.matrix());

  Eigen::Matrix<double, 6, 6> eigen_t = to_eigen(t.matrix());
  Eigen::Matrix<double, 6, 6> eigen_it = to_eigen(t.matrix()).inverse();
  Eigen::Matrix<double, 6, 1> eigen_mv = to_eigen(mv);
  Eigen::Matrix<double, 6, 1> eigen_fv = to_eigen(fv);
  // std::cout << "Eigen FV: " << eigen_fv.transpose() << "\n";
  // std::cout << "Eigen T:\n" << eigen_t << "\n";
  Eigen::Matrix<double, 6, 1> true_tmv = eigen_t * eigen_mv;
  // std::cout << "Eigen transformed MV: " << true_tmv.transpose() << std::endl;
  // Eigen::Matrix<double, 6, 1> true_tfv = eigen_t * eigen_fv;
  // std::cout << "Eigen transformed FV: " << true_tfv.transpose() << std::endl;

  MotionVector tmv = t.apply(mv);
  ForceVector tfv = t.apply(fv);
  // Algebra::print("TDS transformed FV:", tfv);

  Eigen::Matrix<double, 6, 1> true_tmv_i = eigen_it * eigen_mv;
  MotionVector tmv_i = t.apply_inverse(mv);

  Algebra::print("TMV", tmv);
  std::cout << "Eigen: " << true_tmv.transpose() << std::endl;

#if TDS_USE_LEFT_ASSOCIATIVE_TRANSFORMS
  RigidBodyDynamics::Math::SpatialTransform rbdl_t(to_eigen(t.rotation),
                                                   to_eigen(t.translation));
#else
  RigidBodyDynamics::Math::SpatialTransform rbdl_t(
      to_eigen(t.rotation).transpose(), to_eigen(t.translation));
#endif
  RigidBodyDynamics::Math::SpatialVector rbdl_mv = to_rbdl(mv);
  RigidBodyDynamics::Math::SpatialVector rbdl_fv = to_rbdl(fv);

  RigidBodyDynamics::Math::SpatialVector rbdl_t_mv = rbdl_t.apply(rbdl_mv);
  RigidBodyDynamics::Math::SpatialVector rbdl_t_fv =
      rbdl_t.applyTranspose(rbdl_fv);
  std::cout << "RBDL: " << rbdl_t_mv.transpose() << std::endl;

  for (int i = 0; i < 6; ++i) {
    // XXX only RBDL matters
    // EXPECT_NEAR(tmv[i], true_tmv[i], 1e-8);
    // EXPECT_NEAR(tmv_i[i], true_tmv_i[i], 1e-8);
    EXPECT_NEAR(tmv[i], rbdl_t_mv[i], 1e-8);
    EXPECT_NEAR(tfv[i], rbdl_t_fv[i], 1e-8);

    // EXPECT_NEAR(tfv[i], true_tfv[i], 1e-8);
  }
}

template <typename Algebra>
void test_spatial_cross() {
  using Transform = tds::Transform<Algebra>;
  using MotionVector = tds::MotionVector<Algebra>;
  using ForceVector = tds::ForceVector<Algebra>;
  using Vector3 = typename Algebra::Vector3;
  using Matrix6 = typename Algebra::Matrix6;
  using namespace RigidBodyDynamics;

  MotionVector mv1(Vector3(0.1, -0.6, -8.7), Vector3(0.9, 3.8, -0.1));
  MotionVector mv2(Vector3(-5.6, 0.3, 2.8), Vector3(-0.9, 9.3, -13.7));
  ForceVector fv(Vector3(0.9, 3.8, -0.1), Vector3(0.1, -0.6, -8.7));
  MotionVector cross_mv = Algebra::cross(mv1, mv2);
  ForceVector cross_fv = Algebra::cross(mv1, fv);

  Math::SpatialVector rbdl_mv1 = to_rbdl(mv1);
  Math::SpatialVector rbdl_mv2 = to_rbdl(mv2);
  Math::SpatialVector rbdl_fv = to_rbdl(fv);
  Math::SpatialVector rbdl_cross_mv = Math::crossm(rbdl_mv1, rbdl_mv2);
  Math::SpatialVector rbdl_cross_fv = Math::crossf(rbdl_mv1, rbdl_fv);

  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(cross_mv[i], rbdl_cross_mv[i], 1e-8);
    EXPECT_NEAR(cross_fv[i], rbdl_cross_fv[i], 1e-8);
  }
}

template <typename Algebra>
void test_rbi_construction() {
  using Transform = tds::Transform<Algebra>;
  using MotionVector = tds::MotionVector<Algebra>;
  using ForceVector = tds::ForceVector<Algebra>;
  using RBI = tds::RigidBodyInertia<Algebra>;
  using ABI = tds::ArticulatedBodyInertia<Algebra>;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;

  srand(1234);

  Matrix3 I;
  Vector3 com;
  double mass = 4.321;
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      if (j > i) {
        // make sure it is a symmetric matrix
        I(i, j) = I(j, i);
      } else {
        I(i, j) = double(rand()) / RAND_MAX;
      }
    }
    com[j] = double(rand()) / RAND_MAX;
  }

  // Algebra::print("\nI:", I);

  RBI rbi(mass, com, I);

  auto rbdl_rbi = RigidBodyDynamics::Math::SpatialRigidBodyInertia::
      createFromMassComInertiaC(mass, to_eigen(com), to_eigen(I));

  Matrix6 rbi_m = ABI(rbi).matrix();
  // Algebra::print("\nTDS:", rbi_m);
  auto rbdl_rbi_m = rbdl_rbi.toMatrix();
  // std::cout << "\nRBDL:\n" << rbdl_rbi_m << std::endl << std::endl;

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(rbi_m(i, j), rbdl_rbi_m(i, j), 1e-8);
    }
  }
}

template <typename Algebra>
void test_inertia_matrix() {
  using Transform = tds::Transform<Algebra>;
  using MotionVector = tds::MotionVector<Algebra>;
  using ForceVector = tds::ForceVector<Algebra>;
  using ABI = tds::ArticulatedBodyInertia<Algebra>;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;

  srand(1234);

  Matrix3 I, H, M;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      I(i, j) = double(rand()) / RAND_MAX;
      H(i, j) = double(rand()) / RAND_MAX;
      M(i, j) = double(rand()) / RAND_MAX;
    }
  }

  Matrix6 abi = ABI(I, H, M).matrix();

  Matrix3 check;
  Algebra::assign_block(check, abi, 0, 0, 3, 3, 0, 0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(check(i, j), I(i, j), 1e-8);
    }
  }
  Algebra::assign_block(check, abi, 0, 0, 3, 3, 0, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(check(i, j), H(i, j), 1e-8);
    }
  }
  Algebra::assign_block(check, abi, 0, 0, 3, 3, 3, 0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // H^T
      EXPECT_NEAR(check(i, j), H(j, i), 1e-8);
    }
  }
  Algebra::assign_block(check, abi, 0, 0, 3, 3, 3, 3);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(check(i, j), M(i, j), 1e-8);
    }
  }
}

// TEST(Spatial, TransformMatrixConsistency) {
//   test_transform_matrix_consistency<TinyAlgebra<double, TINY::DoubleUtils>>();
//   test_transform_matrix_consistency<tds::EigenAlgebra>();
// }

TEST(Spatial, TransformMatchesRBDL) {
  test_transform_rbdl<TinyAlgebra<double, TINY::DoubleUtils>>();
  test_transform_rbdl<tds::EigenAlgebra>();
}

// TEST(Spatial, TransformInverse) {
//   test_transform_inverse<TinyAlgebra<double, TINY::DoubleUtils>>();
//   test_transform_inverse<tds::EigenAlgebra>();
// }

// TEST(Spatial, TransformTranspose) {
//   test_transform_transpose<TinyAlgebra<double, TINY::DoubleUtils>>();
//   test_transform_transpose<tds::EigenAlgebra>();
// }

// TEST(Spatial, TransformSpatialVector) {
//   // std::cout << "\n\n### TinyAlgebra:\n";
//   test_transform_spatial_vector<TinyAlgebra<double, TINY::DoubleUtils>>();
//   // std::cout << "\n\n### EigenAlgebra:\n";
//   test_transform_spatial_vector<tds::EigenAlgebra>();
// }

// TEST(Spatial, SpatialCross) {
//   std::cout << "\n\n### TinyAlgebra:\n";
//   test_spatial_cross<TinyAlgebra<double, TINY::DoubleUtils>>();
//   std::cout << "\n\n### EigenAlgebra:\n";
//   test_spatial_cross<tds::EigenAlgebra>();
// }

// TEST(Spatial, InertiaFromMassComI) {
//   // std::cout << "\n\n### TinyAlgebra:\n";
//   test_rbi_construction<TinyAlgebra<double, TINY::DoubleUtils>>();
//   // std::cout << "\n\n### EigenAlgebra:\n";
//   test_rbi_construction<tds::EigenAlgebra>();
// }

// TEST(Spatial, InertiaMatrix) {
//   test_inertia_matrix<TinyAlgebra<double, TINY::DoubleUtils>>();
//   test_inertia_matrix<tds::EigenAlgebra>();
// }

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
