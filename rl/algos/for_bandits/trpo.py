n(pi1) = n(pi) + S[s] ro(s) * S[a] pi1(a|s) * (Q-V)
       = n(pi) + S[s] ro(s) * S[a] pi1(a|s) * (r(a,s) - pi1*(s)*r(s))

pi_new(a|s) = (1-a)*pi(a|s) + a*pi1(a|s)

D(p,q) = .5*S |p-q|

Dmax(pi,pi1) = max[s] D(pi(.|s), pi1(.|s))

alpha = Dmax(old,new)

epsi = max[s,a] |Aπ(s, a)|

C = 4*epsi*gamma/(1-gamma)**2

new = arg max[pi] L[old](pi) - CDmax(old,pi)
    = arg max[pi] n(old) S[s] 1/B * S[a] pi(a|s)*A[old](s,a) - 4*epsi*gamma/(1-gamma)**2 * Dmax(old,pi)




maximise loss = n(old) S[s] 1/B * S[a] pi(a|s)*A[old](s,a) - 4*epsi*gamma/(1-gamma)**2 * Dmax(old,pi)