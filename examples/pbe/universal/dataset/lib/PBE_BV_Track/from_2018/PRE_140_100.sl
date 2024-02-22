(set-logic BV)

(define-fun ehad ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (_ BitVec 64))) (_ BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (_ BitVec 64)) (y (_ BitVec 64)) (z (_ BitVec 64))) (_ BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (_ BitVec 64))) (_ BitVec 64)
    ((Start (_ BitVec 64)))
    ((Start (_ BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #x37669312a7bc8098) #x6ecd26254f790130))
(constraint (= (f #x1dda6c43bbe0caea) #x3320900667810180))
(constraint (= (f #xb5eaaee9b88ce5ce) #x6bd55dd37119cb9c))
(constraint (= (f #xe12cb3bec6a3348e) #xc259677d8d46691d))
(constraint (= (f #x12dee2a4711a6e8b) #x25bdc548e234dd17))
(constraint (= (f #x7cb7c0d97865d33e) #xf04f0120e0830478))
(constraint (= (f #x94e580416237e573) #x01820000804f80c4))
(constraint (= (f #xb676d0c4778d23b5) #x48c90100ce100640))
(constraint (= (f #x307ca2abe80e7e17) #x60f94557d01cfc2f))
(constraint (= (f #x3494275e9b9b801c) #x69284ebd37370039))
(constraint (= (f #xe69ca7424d34db8a) #xcd394e849a69b714))
(constraint (= (f #x1ad2408e85b7ad98) #x35a4811d0b6f5b31))
(constraint (= (f #x73272c77e109b2e1) #xc40c10cf80024180))
(constraint (= (f #x03a2377b803b7266) #x06004ce60064c088))
(constraint (= (f #x99c4e0624d9276c2) #x3389c0c49b24ed84))
(constraint (= (f #x4923e8e1987b79a3) #x0007818220e4e204))
(constraint (= (f #x49d5688ec3a5d46e) #x0300801906030098))
(constraint (= (f #xe77aee0ed36d5601) #xcef5dc1da6daac02))
(constraint (= (f #xb2b4a11e642a064a) #x6569423cc8540c94))
(constraint (= (f #x93988bc7870516eb) #x0620070e0c000984))
(constraint (= (f #x8b5c69e2ec35d3eb) #x0430838190430784))
(constraint (= (f #xeb7ec01a7866ee92) #xd6fd8034f0cddd24))
(constraint (= (f #xaa32547cdaeeada6) #x004000f121981208))
(constraint (= (f #xbad93ceee8a2eaba) #x6120719980018060))
(constraint (= (f #xb5c5b082e7b732bd) #x430240018e4c4070))
(constraint (= (f #x1a74127d4ee41b9b) #x34e824fa9dc83737))
(constraint (= (f #xd953654a65c28de4) #x2004800083001380))
(constraint (= (f #xc5c73e0c85e4d6cb) #x8b8e7c190bc9ad97))
(constraint (= (f #x1a242e861a3ee85d) #x34485d0c347dd0bb))
(constraint (= (f #xbb8c1d8b6aae6a48) #x77183b16d55cd490))
(constraint (= (f #xc7593eeead870dee) #x0c207998120c1398))
(constraint (= (f #xa1abedde8e60c50d) #x4357dbbd1cc18a1b))
(constraint (= (f #x378b2b829beecd67) #x4e0406002799108c))
(constraint (= (f #xced51a146ee48741) #x9daa3428ddc90e83))
(constraint (= (f #xec6da904accdb187) #xd8db5209599b630e))
(constraint (= (f #x12732383a9178e6c) #x00c40606000e1890))
(constraint (= (f #x28d8c4eaa27c0b46) #x51b189d544f8168c))
(constraint (= (f #x5d3d5291b2860338) #x3070000240080460))
(constraint (= (f #xe4484ba2dc304829) #x8000060130400000))
(constraint (= (f #x7ca85ec9a4556e0c) #xf950bd9348aadc19))
(constraint (= (f #x3c06be0b227028d9) #x780d7c1644e051b3))
(constraint (= (f #xca9864abed8c12ba) #x0020800792100060))
(constraint (= (f #xedaeb3bdb5d33130) #x9218467243044040))
(constraint (= (f #xc4a16603aa73b35e) #x8942cc0754e766bd))
(constraint (= (f #x435e939e659eee50) #x86bd273ccb3ddca0))
(constraint (= (f #x934d00e63b57d401) #x269a01cc76afa802))
(constraint (= (f #x4b55c5e0ded0eae7) #x040303813901818c))
(constraint (= (f #x41d4ee624b9d9494) #x83a9dcc4973b2929))
(constraint (= (f #xe0eb2d3109297978) #x818410400000e0e0))
(constraint (= (f #x81bc6ea3cdbeb6a6) #x0270980712784808))
(constraint (= (f #xc8986679d91245a3) #x002088e320000204))
(constraint (= (f #x52e9ec70018c73c4) #xa5d3d8e00318e788))
(constraint (= (f #x775b023275dbccbe) #xcc240040c3271078))
(constraint (= (f #x275ae7ba80e433a5) #x0c218e6001804600))
(constraint (= (f #x3b4d217e33ab5beb) #x641000f846042784))
(constraint (= (f #x9de5cb618308e5d6) #x3bcb96c30611cbac))
(constraint (= (f #xece2e5a795e4e4e4) #x9181820e03818180))
(constraint (= (f #x157812e7344c36eb) #x00e0018c40104984))
(constraint (= (f #x06442650be09b6c6) #x0c884ca17c136d8d))
(constraint (= (f #xd600b09db8a87e28) #x080040326000f800))
(constraint (= (f #x5862758e4a5dce8a) #xb0c4eb1c94bb9d15))
(constraint (= (f #x3c34eec7229e208e) #x7869dd8e453c411c))
(constraint (= (f #x708bc8c2712c380d) #xe1179184e258701b))
(constraint (= (f #xe79043ab55ec00eb) #x8e00060403900184))
(constraint (= (f #x837ce12748426e35) #x04f1800c00009840))
(constraint (= (f #x99e8ca53cda5e134) #x2381000712038040))
(constraint (= (f #xc00c3be893e46512) #x801877d127c8ca24))
(constraint (= (f #x23889a0506e5e8ad) #x0600200009838010))
(constraint (= (f #x950e29827210ce1b) #x2a1c5304e4219c37))
(constraint (= (f #x438419391e3075d9) #x870832723c60ebb3))
(constraint (= (f #xe87804e4ea0a2404) #xd0f009c9d4144808))
(constraint (= (f #x1277bdce1e0d0b06) #x24ef7b9c3c1a160d))
(constraint (= (f #x7dca8ece5a175562) #xf3001918200c0080))
(constraint (= (f #xde862a5c327120a8) #x3808003040c00000))
(constraint (= (f #xe7b99823e99ec900) #xcf733047d33d9200))
(constraint (= (f #x73435ed63a52478d) #xe686bdac74a48f1b))
(constraint (= (f #x96eac8bd3c700454) #x2dd5917a78e008a8))
(constraint (= (f #x434b6c9b3e095631) #x0404902478000840))
(constraint (= (f #xe295d7c14182d88a) #xc52baf828305b114))
(constraint (= (f #x1e457748c2a953a9) #x3800cc0100000600))
(constraint (= (f #x454bbe97e14954b7) #x0006780f8000004c))
(constraint (= (f #x279132a043747407) #x4f22654086e8e80f))
(constraint (= (f #x0430557794442000) #x0860aaef28884000))
(constraint (= (f #x3c057c22cbccacbd) #x7000f00107101070))
(constraint (= (f #xec75dcb4c05b6d67) #x90c330410024908c))
(constraint (= (f #xb7263514ebe47d47) #x6e4c6a29d7c8fa8f))
(constraint (= (f #x980258e338c020ee) #x2000218461000198))
(constraint (= (f #x1e7d6854bc60036a) #x38f0800070800480))
(constraint (= (f #x82e7d4c74da32570) #x018f010c120400c0))
(constraint (= (f #x717139272e8ec1b0) #xc0c0600c18190240))
(constraint (= (f #x81cea2ecd8cc61c7) #x039d45d9b198c38f))
(constraint (= (f #xa27c44d68277089e) #x44f889ad04ee113d))
(constraint (= (f #x6509e1c5664eb85c) #xca13c38acc9d70b8))
(constraint (= (f #x2edae5ad09165b51) #x5db5cb5a122cb6a3))
(constraint (= (f #x573008b6bcb93d06) #xae60116d79727a0d))
(constraint (= (f #x99d5b3ceec0ca78a) #x33ab679dd8194f14))
(constraint (= (f #x3be9205e20ee4608) #x77d240bc41dc8c10))
(constraint (= (f #xba2c2ce6a8b2752d) #x601011880040c010))
(constraint (= (f #x0ee4eb40a103d730) #x1981840000070c40))
(constraint (= (f #xee9c2b0994393e30) #x9830040200607840))

(check-synth)

