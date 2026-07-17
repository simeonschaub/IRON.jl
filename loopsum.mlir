module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func @julia_loopsum_68404(%arg0: i64 {llvm.signext}) -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(true) : i1
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bb1(%0, %0 : i64, i64)
  ^bb1(%4: i64, %5: i64):  // 2 preds: ^bb0, ^bb5
    %6 = llvm.icmp "slt" %4, %arg0 : i64
    %7 = llvm.xor %6, %1  : i1
    llvm.cond_br %7, ^bb6, ^bb2
  ^bb2:  // pred: ^bb1
    %8 = llvm.srem %4, %2  : i64
    %9 = llvm.icmp "eq" %8, %0 : i64
    %10 = llvm.xor %9, %1  : i1
    llvm.cond_br %10, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    llvm.br ^bb5(%5 : i64)
  ^bb4:  // pred: ^bb2
    %11 = llvm.add %5, %4  : i64
    llvm.br ^bb5(%11 : i64)
  ^bb5(%12: i64):  // 2 preds: ^bb3, ^bb4
    %13 = llvm.add %4, %3  : i64
    llvm.br ^bb1(%13, %12 : i64, i64)
  ^bb6:  // pred: ^bb1
    llvm.return %5 : i64
  }
}
