#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>
#include <iostream>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>
#include <vector>

namespace dmrgmp {

enum Position {
  LEFT = 0,
  RIGHT = 1,
  WARM = 2,
};

enum BoundaryCondition {
  OBC = 0,
  PBC = 1,
};

using TermSizeList = std::vector<std::pair<int, int>>;
template <typename T> class DMRGSystem {
public:
  using DMRGMatrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit>;
  using DMRGVector = Eigen::Matrix<T, Eigen::Dynamic, 1>; // column vector
  using ElementMatrix = Eigen::Matrix<T, 2, 2, Eigen::RowMajorBit>;
  using DMRGMatVector = std::vector<DMRGMatrix>;
  using DMRGMatVecVector = std::vector<DMRGMatVector>;
  // using DMRGMatVecVector = std::vector<std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit>>>;

  DMRGMatrix deep_copy(DMRGMatrix &other) {
    DMRGMatrix ret = other(Eigen::all, Eigen::all);
    return ret;
  }

  DMRGVector deep_copy(DMRGVector &other) {
    DMRGVector ret = other(Eigen::all, Eigen::all);
    return ret;
  }

  DMRGMatVector deep_copy(DMRGMatVector &other) {
    DMRGMatVector ret;
    for (auto matrix : other) {
      ret.push_back(deep_copy(matrix));
    }
    return ret;
  }

public:
  double psi_dot_psi(DMRGMatrix &psi1, DMRGMatrix &psi2) {
    
    std::complex<double> product(0.0, 0.0);

   
#pragma omp declare	\
    reduction(	\
      complex_sum : \
      std::complex<double> :	\
      omp_out += omp_in )	\
    initializer( omp_priv = omp_orig )

#pragma omp parallel for collapse(2) reduction(complex_sum : product)
    for (int i = 0; i < psi1.rows(); i++) {
      for (int j = 0; j < psi1.cols(); j++) {
        // product += std::conj(psi1(i, j)) * psi2(i, j);
        product = std::conj(psi1(i, j)) * psi2(i, j);
      }
    }

    return product.real();
  }

  void lanczos(DMRGSystem &m, int maxiter, double tol, DMRGMatrix &out_gs,
               double &out_e0, bool use_seed = false,
               bool force_maxiter = false) {
    DMRGMatrix x1 = m.psi;
    DMRGMatrix x2 = m.psi;
    DMRGMatrix gs = m.psi;
    DMRGVector a = DMRGVector::Zero(100);
    DMRGVector b = DMRGVector::Zero(100);
    DMRGMatrix z = DMRGMatrix::Zero(100, 100);

    DMRGMatVector lvectors;
    int control_max = maxiter;

    double e0 = 9999.0;

    if (maxiter == 1) {
      force_maxiter = false;
    }

    out_gs = DMRGMatrix::Zero(gs.rows(), gs.cols());
    if (control_max == 0) {
      out_gs = DMRGMatrix::Identity(1, 1);
      out_e0 = e0;
      m.psi = deep_copy(out_gs);
      m.energy = out_e0;
      return;
    }

    
    b(0) = psi_dot_psi(x1, x1);
    b(0) = std::sqrt(b(0));
    x1 = x1 / b(0);
    b(0) = 1.0;

    int nmax = 99 < maxiter ? 99 : maxiter;

    std::cout << "nmax :" << nmax << std::endl << std::flush;

    for (int iter = 1; iter < nmax + 1; iter++) {
      int eini = e0;
      DMRGMatrix aux;
      if (b[iter - 1] != 0.0) {
        aux = deep_copy(x1);
        x1 = -b[iter - 1] * x2;
        x2 = aux / b[iter - 1];
      }

      std::cout << "check prod\n";

      aux = m.product(x2);

      std::cout << "check \n";

      x1 = x1 + aux;
      a(iter) = psi_dot_psi(x1, x2);
      x1 = x1 - x2 * a[iter];

      b[iter] = psi_dot_psi(x1, x1);
      b[iter] = std::sqrt(b[iter]);
      lvectors.push_back(deep_copy(x2));

   
      z = DMRGMatrix::Zero(iter, iter);

      for (int i = 0; i < iter - 1; i++) {
        z(i, i + 1) = b(i + 1);
        z(i + 1, i) = b(i + 1);
        z(i, i) = a(i + 1);
      }

      z(iter - 1, iter - 1) = a(iter);
      Eigen::ComplexEigenSolver<DMRGMatrix> ces;
      ces.compute(z);

      int col = 0;
      e0 = 9999.0;
      for (int i = 0; i < ces.eigenvalues().size(); i++) {
        auto e = ces.eigenvalues()[i];
        if (e.real() < e0) {
          e0 = e.real();
          col = i;
        }
      }
      e0 = ces.eigenvalues()[col].real();
      out_e0 = e0;

      std::cout << "Iter = " << iter << " Ener = " << e0 << std::endl;

      if (force_maxiter && iter >= control_max ||
          iter >= gs.rows() * gs.cols() || iter == 99 ||
          std::abs(b(iter)) < tol ||
          !force_maxiter && std::abs(eini - e0) <= tol) {
        out_gs = DMRGMatrix::Zero(out_gs.rows(), out_gs.cols());
        for (int n = 0; n < iter; n++) {
          out_gs += ces.eigenvectors()(n, col) * lvectors[n];
        }
        std::cout << "E0 = " << e0 << std::endl;
        maxiter = iter;
        return;
      }
      m.energy = e0;
      out_gs = deep_copy(gs);
      m.psi = deep_copy(out_gs);
    }
  }

public:
  int r1;
  int r2;
  TermSizeList termSzSzlist;

  // Single site operators
  int nsites;
  int nstates;
  int dim_l;
  int dim_r;
  int left_size;
  int right_size;

  int nsweep;
  ElementMatrix sz0;
  ElementMatrix splus0;

  // Useful structures to store the matrices
  DMRGMatVector szL;
  DMRGMatVector szR;
  DMRGMatVector splusL;
  DMRGMatVector splusR;

  DMRGMatVector Ar1L;
  DMRGMatVector Ar1R;
  DMRGMatVector Ar2L;
  DMRGMatVector Ar2R;

  DMRGMatVector Lmtx;
  DMRGMatVector Rmtx;

  DMRGMatrix psi = ElementMatrix::Zero();
  // DMRGMatrix psi = ElementMatrix::Identity();
  DMRGMatrix Ar1 = ElementMatrix::Zero();
  DMRGMatrix Ar2 = ElementMatrix::Zero();
  DMRGMatrix CVreal = ElementMatrix::Zero();
  DMRGMatrix CVimag = ElementMatrix::Zero();
  DMRGMatrix rho = ElementMatrix::Zero();

  Eigen::Matrix<std::complex<double>, 2, 2> correction_vector =
      Eigen::Matrix<std::complex<double>, 2, 2>::Zero();

  double energy = 0.0;
  double error = 0.0;

  DMRGMatrix SziSzj;
  DMRGVector Szi;

  DMRGMatVecVector SzBL, SplusBL, SzBR, SplusBR;

public:
  DMRGSystem(int nsites, int r1, int r2, TermSizeList &termSzSzlist)
      : nsites(nsites), r1(r1), r2(r2), termSzSzlist(termSzSzlist) {
    // initialization
    nstates = 2;
    dim_l = 0;
    dim_r = 0;
    left_size = 0;
    right_size = 0;
    nsweep = 0;

    sz0 = ElementMatrix::Zero();
    splus0 = ElementMatrix::Zero();
    sz0(0, 0) = -0.5;
    sz0(1, 1) = 0.5;
    splus0(1, 0) = 1.0;

    for (int i = 0; i < nsites; i++) {
      ElementMatrix sz0_cp_L = sz0;
      ElementMatrix splus0_cp_L = splus0;
      ElementMatrix sz0_cp_R = sz0;
      ElementMatrix splus0_cp_R = splus0;
      szL.push_back(sz0_cp_L);
      szR.push_back(sz0_cp_R);
      splusL.push_back(splus0_cp_L);
      splusR.push_back(splus0_cp_R);

      Lmtx.push_back(ElementMatrix::Identity());
      Rmtx.push_back(ElementMatrix::Identity());

      Ar1L.push_back(ElementMatrix::Identity());
      Ar1R.push_back(ElementMatrix::Identity());
      Ar2L.push_back(ElementMatrix::Identity());
      Ar2R.push_back(ElementMatrix::Identity());
    }

    // these two are not being used
    SziSzj = DMRGMatrix::Zero(nsites, nsites);
    Szi = DMRGVector::Zero(nsites, 1);

    for (int i = 0; i < termSzSzlist.size(); i++) {
      SzBL.push_back(deep_copy(szL));
      SplusBL.push_back(deep_copy(splusL));
      SzBR.push_back(deep_copy(szR));
      SplusBR.push_back(deep_copy(splusR));
    }
  }

  void build_block_left(int iter, double eta) {
    left_size = iter;
    for (int i = 0; i < termSzSzlist.size(); i++) {
      if (SzBL[i][left_size - 1].rows() > dim_l) {
        dim_l = SzBL[i][left_size - 1].rows();
      }
    }

    DMRGMatrix I_left = DMRGMatrix::Identity(dim_l,dim_l);
    DMRGMatrix I2 = DMRGMatrix::Identity(2,2);

    for (int i = 0; i < termSzSzlist.size(); i++) {
      auto term = termSzSzlist[i];

      if (term.first < left_size && term.second < left_size) {
        SzBL[i][left_size] =
            Eigen::KroneckerProduct(SzBL[i][left_size - 1], I2).eval();
        SplusBL[i][left_size] =
            Eigen::kroneckerProduct(SplusBL[i][left_size - 1], I2).eval();
      } else if (term.first < left_size && term.second == left_size) {
        SzBL[i][left_size] =
            Eigen::KroneckerProduct(SzBL[i][left_size - 1], sz0).eval();
        SplusBL[i][left_size] =
            0.5 * Eigen::KroneckerProduct(SplusBL[i][left_size - 1],
                                          splus0.transpose())
                      .eval() +
            0.5 * Eigen::KroneckerProduct(SplusBL[i][left_size - 1].transpose(),
                                          splus0)
                      .eval();
      } else if (term.first == left_size && term.second > left_size) {
        SzBL[i][left_size] = Eigen::KroneckerProduct(I_left, sz0).eval();
        SplusBL[i][left_size] = Eigen::KroneckerProduct(I_left, splus0).eval();
      } else if (term.first < left_size && term.second > left_size) {
        SzBL[i][left_size] =
            Eigen::KroneckerProduct(SzBL[i][left_size - 1], I2).eval();
        SplusBL[i][left_size] =
            Eigen::KroneckerProduct(SplusBL[i][left_size - 1], I2).eval();
      } else if (term.first > left_size && term.second > left_size) {
        SzBL[i][left_size] = Eigen::KroneckerProduct(I_left, I2).eval();
        SplusBL[i][left_size] = Eigen::KroneckerProduct(I_left, I2).eval();
      } else {
        std::cerr << "WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                     "!!!!!!!!!!!!"
                  << std::endl;
      }

      std::cout << "The " << i << "th term, left_size is " << left_size
                << " SzBL[i] is " <<
          SzBL[i][left_size] << std::endl;
    }
  }

  void build_block_right(int iter, double eta) {
    right_size = iter;
    for (int i = 0; i < termSzSzlist.size(); i++) {
      if (SzBR[i][right_size - 1].rows() > dim_r) {
        dim_r = SzBR[i][right_size - 1].rows();
      }
    }

    DMRGMatrix I_right = DMRGMatrix::Identity(dim_r, dim_r);
    DMRGMatrix I2 = DMRGMatrix::Identity(2,2);

    int mid1 = nsites - 2 - right_size;

    for (int i = 0; i < termSzSzlist.size(); i++) {
      auto term = termSzSzlist[i];

      if (term.first >= mid1 + 2 && term.second >= mid1 + 2) {
        SzBR[i][right_size] =
            Eigen::KroneckerProduct(I2, SzBR[i][right_size - 1]);
        SplusBR[i][right_size] =
            Eigen::KroneckerProduct(I2, SplusBR[i][right_size - 1]);
      } else if (term.first == mid1 + 1 && term.second >= mid1 + 2) {
        SzBR[i][right_size] =
            Eigen::KroneckerProduct(sz0, SzBR[i][right_size - 1]);
        SplusBR[i][right_size] =
            0.5 * Eigen::KroneckerProduct(splus0.transpose(),
                                          SplusBR[i][right_size - 1]) +
            0.5 * Eigen::KroneckerProduct(
                      splus0, SplusBR[i][right_size - 1].transpose());
      } else if (term.first <= mid1 && term.second == mid1 + 1) {
        SzBR[i][right_size] = Eigen::KroneckerProduct(sz0, I_right);
        SplusBR[i][right_size] = Eigen::KroneckerProduct(splus0, I_right);
      } else if (term.first <= mid1 && term.second >= mid1 + 2) {
        SzBR[i][right_size] =
            Eigen::KroneckerProduct(I2, SzBR[i][right_size - 1]);
        SplusBR[i][right_size] =
            Eigen::KroneckerProduct(I2, SplusBR[i][right_size - 1]);
      } else if (term.first <= mid1 && term.second <= mid1) {
        SzBR[i][right_size] = Eigen::KroneckerProduct(I2, I_right);
        SplusBR[i][right_size] = Eigen::KroneckerProduct(I2, I_right);
      } else {
        std::cerr << "WRONG "
                     "right!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                     "!!!!!!!!!!!!"
                  << std::endl;
      }
      std::cout << "The " << i << "th term, right_size is " << right_size
                << " SzBR[i] is " <<
          SzBR[i][right_size] << std::endl;
    }
  }

  void ground_state(Position position) {
    for (int i = 0; i < termSzSzlist.size(); i++) {
      if (SzBL[i][left_size].rows() > dim_l) {
        dim_l = SzBL[i][left_size].rows();
      }
    }

    for (int i = 0; i < termSzSzlist.size(); i++) {
      if (SzBR[i][right_size].rows() > dim_r) {
        dim_r = SzBR[i][right_size].rows();
      }
    }

    /*     if(nsweep >= 1) {
          if (position == Position::LEFT && left_size >= 2) {
              DMRGMatrix L_mtx = deep_copy(Lmtx[left_size]);


          } else if (position == Position::RIGHT && right_size >= 2) {

          }
        } */

    std::cout << "psi before: " << psi.rows() << "," << psi.cols() << std::endl;
    psi.resize(dim_l, dim_r);
    std::cout << "psi after: " << psi.rows() << "," << psi.cols() << std::endl;

    
    int maxiter = dim_l * dim_r;

    std::cout << "maxiter: " << maxiter << std::endl;

    DMRGMatrix out_gs;
    double out_e0;
    if (nsweep > 1 && position == Position::WARM) {
      lanczos(*this, maxiter, 1.e-7, out_gs, out_e0);
    } else {
      lanczos(*this, maxiter, 1.e-7, out_gs, out_e0, false);
    }

    // energy = out_e0;

    std::cout << "The solved psi is: " << out_gs.rows() << "," << out_gs.cols()
              << std::endl;
    std::cout << "The solved energy is: " << out_e0 << std::endl;
    
  }

  void density_matrix_warmup(Position position) {
    if (position == Position::LEFT) {
      rho = psi * psi.transpose().conjugate();
    } else {
      rho = psi.transpose().conjugate() * psi;
    }
  }

  void density_matrix(Position position) {
    if (position == Position::LEFT) {
      rho = psi * psi.transpose().conjugate();
    } else {
      rho = psi.transpose() * psi.conjugate();
    }
  }

  void truncate(Position position, int m_bond) {
    Eigen::SelfAdjointEigenSolver<DMRGMatrix> ces;
    ces.compute(rho);
    auto rho_eig = ces.eigenvalues();
    auto rho_evec = ces.eigenvectors();

    nstates = m_bond;

    std::cout << "The real dim of RHO is " << rho_eig.size()
              << ", the dimention of U is " << rho_evec.rows() << ","
              << rho_evec.cols() << std::endl;

    std::vector<std::pair<double, int>> rho_eig_real;
    for (int i = 0; i < rho_eig.size(); i++) {
      rho_eig_real.push_back(std::make_pair(rho_eig[i], i));
    }

   
    std::sort(rho_eig_real.begin(), rho_eig_real.end());

    double error = 0.0;

    if (m_bond < rho_eig.size()) {
      for (int i = 0; i < rho_eig.size() - m_bond; i++) {
        error += rho_eig[rho_eig_real[i].second];
      }
    }

    std::cout << "Truncation err = " << error << std::endl;

    DMRGMatrix aux = rho_evec;


    if (rho.rows() > m_bond) {
      aux.resize(aux.rows(), m_bond);
      for (int i = rho_eig_real.size() - 1, n = 0;
           i > rho_eig_real.size() - 1 - m_bond; i--, n++) {
        aux.col(n) = rho_evec.col(rho_eig_real[i].second);
      }
    }
    rho_evec = deep_copy(aux);

    DMRGMatrix U = rho_evec.transpose().conjugate();

    std::cout << "Before rotation: " << U.rows() << "," << U.cols() << ","
              << psi.rows() << "," << psi.cols() << std::endl;

    if (position == Position::LEFT) {
      for (int i = 0; i < termSzSzlist.size(); i++) {
        auto term = termSzSzlist[i];
        auto aux2 = SplusBL[i][left_size] * rho_evec;
        SplusBL[i][left_size] = U * aux2;
        auto aux3 = SzBL[i][left_size] * rho_evec;
        SzBL[i][left_size] = U * aux3;
      }
    } else {
      int mid1 = nsites - 2 - right_size;
      for (int i = 0; i < termSzSzlist.size(); i++) {
        auto term = termSzSzlist[i];
        auto aux2 = SplusBR[i][right_size] * rho_evec;
        SplusBR[i][right_size] = U * aux2;
        auto aux3 = SzBR[i][right_size] * rho_evec;
        SzBR[i][right_size] = U * aux3;
      }
    }
  }

  DMRGMatrix product(DMRGMatrix &psi) {
    DMRGMatrix npsi = DMRGMatrix::Zero(psi.rows(), psi.cols());
    int mid1 = nsites - 2 - right_size;

    for (int i = 0; i < termSzSzlist.size(); i++) {
      auto term = termSzSzlist[i];

      if (term.first <= left_size - 1 && term.second <= left_size - 1 ||
          term.first <= left_size - 1 && term.second == left_size) {
        // only LEFT dot |psi>
        std::cout << "print prod 1 0\n";
        std::cout <<  SzBL[i][left_size].rows() << "," << SzBL[i][left_size].cols() << "," << psi.rows() << "," << psi.cols() << "\n";
        npsi += SzBL[i][left_size] * psi;
        std::cout << "print prod 1\n";
        npsi += SplusBL[i][left_size] * psi;
      } else if (term.first >= mid1 + 2 && term.second >= mid1 + 2 ||
                 term.first == mid1 + 1 && term.second >= mid1 + 2) {
        // only |psi> RIGHT dot
        std::cout << "print prod 2 0\n";
        npsi += psi * SzBL[i][right_size].transpose();
        std::cout << "print prod 2\n";
        npsi += psi * SplusBR[i][right_size].transpose();
      } else if (term.first == left_size && term.second == mid1 + 1 ||
                 term.first <= left_size - 1 && term.second == mid1 + 1 ||
                 term.first == left_size && term.second >= mid1 + 2 ||
                 term.first <= left_size - 1 && term.second >= mid1 + 2) {
        // LEFT dot |psi> RIGHT dot
        // Sz.Sz
        std::cout << "print prod 3\n";
        auto tmat = psi * SzBR[i][right_size].transpose();
        npsi += SzBL[i][left_size] * tmat;
        // S+.S-
        auto tmat2 = psi * SplusBR[i][right_size] * 0.5;
        npsi += SplusBL[i][left_size] * tmat2;
        // S-,S+
        auto tmat3 = psi * SplusBR[i][right_size].transpose() * 0.5;
        npsi += SplusBL[i][left_size].transpose() * tmat3;
      } else {
        if (nsweep > 0) {
          std::cout << "Tell me the if else case is "
                       "wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    << std::endl;
        }
      }
    }
    // npsi.eval(); // this is not necessary since psi += psi + f();
    return npsi;
  }
};

} // namespace dmrgmp