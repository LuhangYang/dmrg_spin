#include "dmrg.h"
#include <complex>
#include <iostream>

int main() {
  std::cout << "hello dmrg" << std::endl;

  dmrgmp::TermSizeList l{{0, 1}, {1, 0}};

  // init parameters
  int Lx = 8;
  int Ly = 1;
  int nsites = Lx * Ly;
  int n_states_to_keep = 16;
  int n_sweeps = 3;
  double eta = 0.05;

  dmrgmp::BoundaryCondition xBC = dmrgmp::BoundaryCondition::PBC;
  dmrgmp::BoundaryCondition yBC = dmrgmp::BoundaryCondition::OBC;

  dmrgmp::TermSizeList termSzSzlist;

  for (int x = 0; x < Lx - 1; x++) {
    for (int y = 0; y < Ly - 1; y++) {
      termSzSzlist.push_back(std::make_pair(x * Ly + y, (x + 1) * Ly + y));
      termSzSzlist.push_back(std::make_pair(x * Ly + y, x * Ly + y + 1));
    }
  }

  for (int x = 0; x < Lx - 1; x++) {
    termSzSzlist.push_back(
        std::make_pair(x * Ly + Ly - 1, (x + 1) * Ly + Ly - 1));
    // std::cout << x * Ly + Ly - 1 << "|" << (x + 1) * Ly + Ly - 1 <<
    // std::endl;
  }

  for (int y = 0; y < Ly - 1; y++) {
    termSzSzlist.push_back(
        std::make_pair((Lx - 1) * Ly + y, (Lx - 1) * Ly + y + 1));
  }

  if (yBC == dmrgmp::BoundaryCondition::PBC) {
    for (int x = 0; x < Lx; x++) {
      termSzSzlist.push_back(std::make_pair(x * Ly, x * Ly + (Ly - 1)));
    }
  }

  if (xBC == dmrgmp::BoundaryCondition::PBC) {
    for (int y = 0; y < Ly; y++) {
      termSzSzlist.push_back(std::make_pair(y, (Lx - 1) * Ly + y));
    }
  }

  // print terms
  for (int i = 0; i < termSzSzlist.size(); i++) {
    auto iter = termSzSzlist[i];
    std::cout << iter.first << "," << iter.second << std::endl;
  }

  for (int r1 = 0; r1 < 1; r1++) {
    std::cout << "NOW WORKING ON THE " << r1
              << " SITE************************************************"
              << std::endl;
    for (int r2 = 0; r2 < 1; r2++) {
      dmrgmp::DMRGSystem S = dmrgmp::DMRGSystem<std::complex<double>>(
          nsites, r1, r2, termSzSzlist);
      std::cout << "initial term list: " << S.termSzSzlist.size() << std::endl;

      for (int iter = 1; iter < nsites / 2; iter++) {
        std::cout << "WARMUP ITERATION " << iter << "," << S.dim_l << ","
                  << S.dim_r << std::endl;
        S.build_block_left(iter, eta);

        S.build_block_right(iter, eta);

        S.ground_state(dmrgmp::Position::WARM);

        S.density_matrix_warmup(dmrgmp::Position::LEFT);

        S.truncate(dmrgmp::Position::LEFT, n_states_to_keep);

        S.density_matrix_warmup(dmrgmp::Position::RIGHT);
        S.truncate(dmrgmp::Position::RIGHT, n_states_to_keep);
      }
      int first_iter = nsites / 2;
      for (int sweep = 1; sweep < n_sweeps; sweep++) {
        std::cout << "NEW SWEEP: " << sweep << std::endl;
        for (int iter = first_iter; iter < nsites - 3; iter++) {
          std::cout << "LEFT-TO-RIGHT ITERATION " << iter << S.dim_l << S.dim_r << std::endl;
          S.build_block_left(iter, eta);
          S.build_block_right(nsites - iter - 2, eta);
          S.ground_state(dmrgmp::Position::LEFT);
          if (sweep > 1) {
            S.density_matrix(dmrgmp::Position::LEFT);
          } else {
            S.density_matrix_warmup(dmrgmp::Position::LEFT);
          }

          S.truncate(dmrgmp::Position::LEFT, n_states_to_keep);
        }

        first_iter = 1;
        for (int iter = first_iter; iter < nsites - 3; iter++) {
          std::cout << "RIGHT-TO-LEFT ITERATION " << iter << S.dim_l << S.dim_r << std::endl;
          S.build_block_right(iter, eta);
          S.build_block_left(nsites-iter-2,eta);
          S.ground_state(dmrgmp::Position::RIGHT);

          if(sweep > 1) {
            S.density_matrix(dmrgmp::Position::RIGHT);
          }else {
            S.density_matrix_warmup(dmrgmp::Position::RIGHT);
          }
          S.truncate(dmrgmp::Position::RIGHT,n_states_to_keep);
        }

      S.nsweep += 1;
      }
    }
  }

  
}