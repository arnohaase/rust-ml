use std::cell::RefCell;
use triomphe::Arc;

pub struct Collection<const N: usize, X> {
    buf: Arc<RefCell<Vec<X>>>,
}
impl <const N: usize, X> Collection<N, X> {
    pub fn new(data: Vec<X>) -> Collection<N, X> { //TODO from iter
        assert_eq!(N, data.len());
        Collection { buf: Arc::new(RefCell::new(data)) }
    }
}
