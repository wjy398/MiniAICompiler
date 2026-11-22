module {
  func.func @basic(%a: f32, %b: f32) -> f32 {
    %two = arith.constant 2.0 : f32
    %three = arith.constant 3.0 : f32
    %five = arith.addf %two, %three : f32
    %x = arith.addf %a, %b : f32
    %duplicate = arith.addf %a, %b : f32
    %dead = arith.mulf %a, %b : f32
    %sum = arith.addf %x, %duplicate : f32
    %result = arith.addf %sum, %five : f32
    return %result : f32
  }
}
