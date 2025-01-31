(set-logic BV)
(define-fun ehad ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (_ BitVec 64))) (_ BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun if0 ((x (_ BitVec 64)) (y (_ BitVec 64)) (z (_ BitVec 64))) (_ BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (_ BitVec 64))) (_ BitVec 64)
    ((Start (_ BitVec 64)))
    ((Start (_ BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (if0 Start Start Start)))))

(constraint (= (f #x6d7548e7151cdb31) #x00006d7548e7151c))
(constraint (= (f #xab1d14268c59c247) #x563a284d18b3848e))
(constraint (= (f #x376e39db35c0b106) #x0000376e39db35c0))
(constraint (= (f #x6927d90ed5e71d50) #x00006927d90ed5e7))
(constraint (= (f #x16e1ad0e43d5368e) #x000016e1ad0e43d5))
(constraint (= (f #x88087e37c8e76e9e) #x000088087e37c8e7))
(constraint (= (f #x6004048936e20cc1) #x00006004048936e2))
(constraint (= (f #xc7b21e9ca69d8aeb) #x8f643d394d3b15d6))
(constraint (= (f #xb1243dd9e0c85168) #x0000b1243dd9e0c8))
(constraint (= (f #xae9d62510e3515aa) #x0000ae9d62510e35))
(constraint (= (f #x3dee35a763a8ebeb) #x7bdc6b4ec751d7d6))
(constraint (= (f #xcc4daa8838174eec) #x0000cc4daa883817))
(constraint (= (f #x2a7c4c3e909ecc39) #x00002a7c4c3e909e))
(constraint (= (f #xb9e5177e3e31032b) #x73ca2efc7c620656))
(constraint (= (f #x5a2bdceb9d9e40de) #x00005a2bdceb9d9e))
(constraint (= (f #x81e8ce9a6c7b9c74) #x000081e8ce9a6c7b))
(constraint (= (f #xe3728e4d899cc090) #x0000e3728e4d899c))
(constraint (= (f #xee66eaabb897e7e7) #xdccdd557712fcfce))
(constraint (= (f #x581dee34296c4904) #x0000581dee34296c))
(constraint (= (f #x2947da4497e7a2ac) #x00002947da4497e7))
(constraint (= (f #x8139947893be71ec) #x00008139947893be))
(constraint (= (f #x484e04687123478b) #x909c08d0e2468f16))
(constraint (= (f #xccd354c255916411) #x0000ccd354c25591))
(constraint (= (f #xe566c68d8a825969) #x0000e566c68d8a82))
(constraint (= (f #xe78a83279d389cee) #x0000e78a83279d38))
(constraint (= (f #x9dee65e275bb4c43) #x3bdccbc4eb769886))
(constraint (= (f #xeed83ce1d0a38e56) #x0000eed83ce1d0a3))
(constraint (= (f #xcc5ea091814ad37a) #x0000cc5ea091814a))
(constraint (= (f #x46aca595e93ce31d) #x000046aca595e93c))
(constraint (= (f #x98466be3e1984ee7) #x308cd7c7c3309dce))
(constraint (= (f #x5100291966b06c49) #x00005100291966b0))
(constraint (= (f #x859aeac4d5ea7e72) #x0000859aeac4d5ea))
(constraint (= (f #xdae867601caee49a) #x0000dae867601cae))
(constraint (= (f #x7e5a3ec2e28a73ed) #x00007e5a3ec2e28a))
(constraint (= (f #x58e119b20e68b7c7) #xb1c233641cd16f8e))
(constraint (= (f #x1513e50252013935) #x00001513e5025201))
(constraint (= (f #x796c7484ae94543b) #xf2d8e9095d28a876))
(constraint (= (f #x88421caa2ddb4bd0) #x000088421caa2ddb))
(constraint (= (f #xe02c8ada3901eecc) #x0000e02c8ada3901))
(constraint (= (f #xee87098618741871) #x0000ee8709861874))
(constraint (= (f #x2e5e362d15eee351) #x00002e5e362d15ee))
(constraint (= (f #x0e15441d63ed9462) #x00000e15441d63ed))
(constraint (= (f #x88b2dc8e88729c82) #x000088b2dc8e8872))
(constraint (= (f #x411a79cdae41bd16) #x0000411a79cdae41))
(constraint (= (f #xbececc24e7952095) #x0000bececc24e795))
(constraint (= (f #x0eb46ae8965eaea0) #x00000eb46ae8965e))
(constraint (= (f #x4e46ece0234eca5c) #x00004e46ece0234e))
(constraint (= (f #x903e544c0d514dee) #x0000903e544c0d51))
(constraint (= (f #xaeb207eeec73ea8e) #x0000aeb207eeec73))
(constraint (= (f #x5a25e62219d186a1) #x00005a25e62219d1))
(constraint (= (f #x0e77ee3365ed53e6) #x00000e77ee3365ed))
(constraint (= (f #x0ee5574c2564c61a) #x00000ee5574c2564))
(constraint (= (f #x4e03ee9c62570054) #x00004e03ee9c6257))
(constraint (= (f #xee3e0403b15c4e8d) #x0000ee3e0403b15c))
(constraint (= (f #x48e639bdbecaa16d) #x000048e639bdbeca))
(constraint (= (f #xe028c05915427d4e) #x0000e028c0591542))
(constraint (= (f #xde989340433a134e) #x0000de989340433a))
(constraint (= (f #xe4dc120252b579e2) #x0000e4dc120252b5))
(constraint (= (f #x2e21394e9cacb6ed) #x00002e21394e9cac))
(constraint (= (f #x957b736563ceb33c) #x0000957b736563ce))
(constraint (= (f #xeae4a4982076ee7e) #x0000eae4a4982076))
(constraint (= (f #xa0590c6d03ade835) #x0000a0590c6d03ad))
(constraint (= (f #x2c11e205d6580eec) #x00002c11e205d658))
(constraint (= (f #xa6cb78b71ca5e010) #x0000a6cb78b71ca5))
(constraint (= (f #x1d2e354524ad0ecb) #x3a5c6a8a495a1d96))
(constraint (= (f #xcb329ce230e947d8) #x0000cb329ce230e9))
(constraint (= (f #x6208276481d8de04) #x00006208276481d8))
(constraint (= (f #xe768e48aaa445d2b) #xced1c9155488ba56))
(constraint (= (f #xda01e2dea5dc4db8) #x0000da01e2dea5dc))
(constraint (= (f #x456bc3395e481dcb) #x8ad78672bc903b96))
(constraint (= (f #x4dae231b0e5e96ec) #x00004dae231b0e5e))
(constraint (= (f #x0224ba55d3727ed0) #x00000224ba55d372))
(constraint (= (f #x264e55411791c6ed) #x0000264e55411791))
(constraint (= (f #x03b9ce35233de893) #x07739c6a467bd126))
(constraint (= (f #x3829b15b23bab5b7) #x705362b647756b6e))
(constraint (= (f #x67ba09a8c0069dd2) #x000067ba09a8c006))
(constraint (= (f #x5dc5e1430aa2ad4e) #x00005dc5e1430aa2))
(constraint (= (f #xe4c2de291d92a0ee) #x0000e4c2de291d92))
(constraint (= (f #xada7d182d3362944) #x0000ada7d182d336))
(constraint (= (f #x5a14908c850b0d30) #x00005a14908c850b))
(constraint (= (f #xe6c3a55edaca98e5) #x0000e6c3a55edaca))
(constraint (= (f #x2ee2d53d05215a84) #x00002ee2d53d0521))
(constraint (= (f #xe1cc71aae6ede9d7) #xc398e355cddbd3ae))
(constraint (= (f #x08764de82cab1903) #x10ec9bd059563206))
(constraint (= (f #x5ded67e40592a552) #x00005ded67e40592))
(constraint (= (f #x6d70b065ee3bbb34) #x00006d70b065ee3b))
(constraint (= (f #xb49181019e57081c) #x0000b49181019e57))
(constraint (= (f #x4bd7de90523c2ea0) #x00004bd7de90523c))
(constraint (= (f #x3a2e5b99ee7bdc03) #x745cb733dcf7b806))
(constraint (= (f #x8c10b9e618147ba3) #x182173cc3028f746))
(constraint (= (f #x22e5a4a7e6ea39d0) #x000022e5a4a7e6ea))
(constraint (= (f #x71951ed8d12943e3) #xe32a3db1a25287c6))
(constraint (= (f #x59621d54ee51e744) #x000059621d54ee51))
(constraint (= (f #x0536a7b2b7c68c98) #x00000536a7b2b7c6))
(constraint (= (f #x897733de57cb38c3) #x12ee67bcaf967186))
(constraint (= (f #xe9e93987e32390da) #x0000e9e93987e323))
(constraint (= (f #x58e64a0e034b1e89) #x000058e64a0e034b))
(constraint (= (f #x2d496197501191cc) #x00002d4961975011))
(constraint (= (f #xb39e19462499c80c) #x0000b39e19462499))
(constraint (= (f #x6804b1633abede26) #x00006804b1633abe))

(check-synth)

