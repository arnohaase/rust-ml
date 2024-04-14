

#[cfg(test)]
mod test {
    #[test]
    fn test_blas() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];

        unsafe {
            blas::daxpy(3, 1.5, &x, 1, &mut y, 1);
        }

        println!("{:?}", y);
    }
}