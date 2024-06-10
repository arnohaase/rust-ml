
pub trait ElementWise {
    fn plus(&self, other: &Self) -> Self;
    fn minus(&self, other: &Self) -> Self;
    fn mult(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;

    fn plus_equal(&mut self, other: &Self);
    fn minus_equal(&mut self, other: &Self);
    fn mult_equal(&mut self, other: &Self);
    fn div_equal(&mut self, other: &Self);
}
