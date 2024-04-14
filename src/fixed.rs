


#[derive(Clone)]
struct Vector<const N: usize> {
    raw: [f64;N],
}
impl <const N: usize> Vector<N> {
    fn zero() -> Vector<N> {
        Vector {
            raw: [0.0; N],
        }
    }

    fn compute(f: impl Fn(usize) -> f64) -> Vector<N> {
        let mut raw = [0.0; N];
        for i in 0..N {
            raw[i] = f(i);
        }
        Vector { raw }
    }

    /// self += alpha * x
    ///
    /// This corresponds with the `daxpy` BLAS function
    fn plus_alpha_x(&mut self, alpha: f64, x: &Vector<N>) {
        unsafe {
            blas::daxpy(3, alpha, &x.raw, 1, &mut self.raw, 1);
        }
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> Vector<N> {
        let mut raw = [0.0;N];
        for i in 0..N {
            raw[i] = f(self.raw[i]);
        }
        Vector { raw }
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;
    use crate::fixed::Vector;

    #[test]
    fn test_no_vector() {
        let mut a = 0.1;
        let mut b = 0.9;
        let mut c = 0.1;
        let mut d = -0.1;

        // data.push(F::random_0_1() * (max-min) + min);

        let x = Vector::<5_000>::compute(|_| rand::random::<f64>() * 2.0 * PI - PI);
        let y = x.map(|x| x.sin());

        let learning_rate = 1e-6;

        for t in 0..2_000 {
            let y_pred = x.map(|x| a + b*x + c*x.powi(2) + d*x.powi(3));

            // loss = sum ( (y - a - bx - cx² - dx³)^2 )

            let mut grad_a= 0.;
            let mut grad_b= 0.;
            let mut grad_c= 0.;
            let mut grad_d= 0.;

            let mut loss = 0.;



            for i in 0..x.raw.len() {
                grad_a += 2.*(y_pred.raw[i] - y.raw[i]);
                grad_b += 2.*(y_pred.raw[i] - y.raw[i]) * x.raw[i];
                grad_c += 2.*(y_pred.raw[i] - y.raw[i]) * x.raw[i].powi(2);
                grad_d += 2.*(y_pred.raw[i] - y.raw[i]) * x.raw[i].powi(3);

                loss += (y.raw[i] - y_pred.raw[i]).powi(2);
            }
            a -= grad_a * learning_rate;
            b -= grad_b * learning_rate;
            c -= grad_c * learning_rate;
            d -= grad_d * learning_rate;

            if t%100 == 0 {
                println!("loss: {loss}  a: {a}  b: {b}  c: {c}  d: {d}");
            }
        }
    }
}

