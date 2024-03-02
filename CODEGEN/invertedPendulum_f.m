function xdot = invertedPendulum_f(in1,F,in3)
%INVERTEDPENDULUM_F
%    XDOT = INVERTEDPENDULUM_F(IN1,F,IN3)

%    This function was generated by the Symbolic Math Toolbox version 8.3.
%    09-Jan-2020 11:59:40

I = in3(3,:);
Mc = in3(1,:);
Mp = in3(2,:);
b = in3(6,:);
ds = in1(2,:);
dth = in1(4,:);
g = in3(4,:);
l = in3(5,:);
th = in1(3,:);
t2 = cos(th);
t3 = sin(th);
t4 = Mp.^2;
t5 = dth.^2;
t6 = l.^2;
t7 = th.*2.0;
t8 = I.*Mc.*4.0;
t9 = I.*Mp.*4.0;
t10 = Mc.*Mp.*t6;
xdot = [ds;(F.*I.*8.0+F.*Mp.*t6.*2.0-I.*b.*ds.*8.0+l.*t3.*t5.*t9+l.^3.*t3.*t4.*t5+g.*t4.*t6.*sin(t7)-Mp.*b.*ds.*t6.*2.0)./(t8.*2.0+t9.*2.0+t10.*2.0-t4.*t6.*cos(t7).*2.0);dth;(Mp.*l.*(F.*t2.*2.0+Mc.*g.*t3+Mp.*g.*t3-b.*ds.*t2.*2.0+Mp.*l.*t2.*t3.*t5).*-2.0)./(t8+t9+t10-t4.*t6.*(t2.^2.*2.0-1.0))];
