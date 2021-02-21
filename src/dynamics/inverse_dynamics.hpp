#pragma once

#include <cmath>

#include "../multi_body.hpp"
#include "kinematics.hpp"
#include "math/conditionals.hpp"

namespace tds {
/**
 * Computes the joint torques to achieve the given joint velocities and
 * accelerations. To compute gravity compensation terms, set qd, qdd to zero
 * and pass in negative gravity. To compute only the Coriolis and centrifugal
 * terms, set gravity, qdd and external forces to zero while keeping the joint
 * velocities qd unchanged.
 * @param q Joint positions.
 * @param qd Joint velocities.
 * @param qdd Joint accelerations.
 * @param gravity Gravity.
 * @param tau Joint forces (output).
 */
template <typename Algebra>
void inverse_dynamics(MultiBody<Algebra> &mb,
                      const typename Algebra::VectorX &q,
                      const typename Algebra::VectorX &qd,
                      const typename Algebra::VectorX &qdd,
                      const typename Algebra::Vector3 &gravity,
                      typename Algebra::VectorX &tau) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using VectorX = typename Algebra::VectorX;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  using Quaternion = typename Algebra::Quaternion;
  typedef tds::Transform<Algebra> Transform;
  typedef tds::MotionVector<Algebra> MotionVector;
  typedef tds::ForceVector<Algebra> ForceVector;
  typedef tds::Link<Algebra> Link;
  typedef tds::RigidBodyInertia<Algebra> RigidBodyInertia;
  typedef tds::ArticulatedBodyInertia<Algebra> ArticulatedBodyInertia;

  assert(Algebra::size(q) == mb.dof());
  assert(Algebra::size(qd) == mb.dof_qd());
  assert(Algebra::size(qdd) == mb.dof_qd());
  assert(Algebra::size(tau) == mb.dof_actuated());

  MotionVector spatial_gravity;
  spatial_gravity.bottom = gravity;

  // in the following, the variable names for articulated terms I^A, p^A are
  // used for composite rigid body terms I^c, p^c to avoid introducing more
  // variables

  m_baseAcceleration = spatial_gravity;
  forward_kinematics(q, qd, qdd);

  if (!m_isFloating) {
    int tau_index = m_dof - 1;
    for (int i = static_cast<int>(m_links.size() - 1); i >= 0; i--) {
      TinyLink &link = m_links[i];
      int parent = link.m_parent_index;
      if (link.m_joint_type != JOINT_FIXED) {
        tau[tau_index] = link.m_S.dot(link.m_f);
        --tau_index;
      }
      if (parent >= 0) {
        m_links[parent].m_f += link.m_X_parent2.apply_transpose(link.m_f);
      }
    }
    return;
  }

  // I_0^c, p_0^c are (partially) computed by forward_kinematics
  m_baseBiasForce += m_baseInertia.mul_inv(m_baseAcceleration);

  for (int i = static_cast<int>(m_links.size() - 1); i >= 0; i--) {
    TinyLink &link = m_links[i];
    int parent = link.m_parent_index;
    TinySymmetricSpatialDyad &parent_Ic =
        parent >= 0 ? m_links[parent].m_IA : m_baseArticulatedInertia;
    // forward kinematics computes composite rigid-body bias force p^c as f
    TinySpatialMotionVector &parent_pc =
        parent >= 0 ? m_links[parent].m_f : m_baseBiasForce;
    parent_Ic += TinySymmetricSpatialDyad::shift(link.m_IA, link.m_X_parent2);
    parent_pc += link.m_X_parent2.apply_transpose(link.m_f);
  }

  m_baseAcceleration =
      -m_baseArticulatedInertia.inverse().mul_inv(m_baseBiasForce);

  int tau_index = 0;
  for (int i = 0; i < static_cast<int>(m_links.size()); i++) {
    TinyLink &link = m_links[i];
    //!!! The implementation is different from Featherstone Table 9.6, the
    //!!! commented-out lines correspond to the book implementation that
    //!!! leads to forces too low to compensate gravity in the joints (see
    //!!! gravity_compensation target).
    //      int parent = link.m_parent_index;
    //      const TinySpatialMotionVector& parent_a =
    //          (parent >= 0) ? m_links[parent].m_a : m_baseAcceleration;
    //
    //      link.m_a = link.m_X_parent2.apply(parent_a);

    if (link.m_joint_type != JOINT_FIXED) {
      tau[tau_index] = link.m_S.dot(
          link.m_f);  // link.m_S.dot(link.m_IA.mul_inv(link.m_a) + link.m_f);
      ++tau_index;
    }
  }
}  // namespace tds