(set-logic LIA)

(synth-fun inv ((x Int) (y Int) (l Int)) Bool
    ((Start Bool) (AtomicFormula Bool) (Sum Int) (Term Int) (Sign Int) (Var Int) (Const Int))
    ((Start Bool ((and AtomicFormula AtomicFormula) (or AtomicFormula AtomicFormula)))
    (AtomicFormula Bool ((<= Sum Const) (= Sum Const)))
    (Sum Int ((+ Term Term)))
    (Term Int ((* Sign Var)))
    (Sign Int (0 1 -1))
    (Var Int (x y l))
    (Const Int (-3 -2 -1 0 1 2 3))))

(define-fun implies ((b1 Bool) (b2 Bool)) Bool
    (or (not b1) b2))
(define-fun and3 ((b1 Bool) (b2 Bool) (b3 Bool)) Bool
    (and (and b1 b2) b3))
(define-fun and4 ((b1 Bool) (b2 Bool) (b3 Bool) (b4 Bool)) Bool
    (and (and3 b1 b2 b3) b4))
(define-fun and5 ((b1 Bool) (b2 Bool) (b3 Bool) (b4 Bool) (b5 Bool)) Bool
    (and (and4 b1 b2 b3 b4) b5))
(define-fun and6 ((b1 Bool) (b2 Bool) (b3 Bool) (b4 Bool) (b5 Bool) (b6 Bool)) Bool
    (and (and5 b1 b2 b3 b4 b5) b6))
(define-fun or3 ((b1 Bool) (b2 Bool) (b3 Bool)) Bool
    (or (or b1 b2) b3))
(define-fun or4 ((b1 Bool) (b2 Bool) (b3 Bool) (b4 Bool)) Bool
    (or (or3 b1 b2 b3) b4))
(define-fun or5 ((b1 Bool) (b2 Bool) (b3 Bool) (b4 Bool) (b5 Bool)) Bool
    (or (or4 b1 b2 b3 b4) b5))
(declare-var x Int)
(declare-var y Int)
(declare-var l Int)
(declare-var x1 Int)
(declare-var y1 Int)
(declare-var l1 Int)
(declare-var l2 Int)
(declare-var l3 Int)
(constraint (implies (and4 (= l1 0) (= l2 1) (= x y) (or (and (= l3 0) (= y1 (+ y 1))) (and (= l3 l2) (= y1 y)))) (inv x y1 l3)))
(constraint (implies (and5 (inv x y l) (not (= x y)) (= l1 1) (= x1 y) (or (and (= l2 0) (= y1 (+ y 1))) (and (= l2 l1) (= y1 y)))) (inv x1 y1 l2)))
(constraint (implies (and (inv x y l) (= x y)) (= l 1)))

(check-synth)

