!------program for trajectory calculation in SF------
!------based on principle of least action path------- 

 module inputs     
  implicit real*8 (a-h,o-z)
  double precision,dimension(:),  allocatable :: q20,q22
  double precision,dimension(:,:),allocatable :: b1,b2,b3,v,e0
  integer iq20max,iq22max
 end module inputs

 program main
 use inputs
 implicit real*8 (a-h,o-z)
 integer, dimension(:,:), allocatable :: ip
 integer, dimension(:)  , allocatable :: iend,jend
 double precision, dimension(:  ), allocatable :: q0nd,q2nd,send
 double precision, dimension(:,:), allocatable :: s
 double precision, dimension(:  ), allocatable :: a
 
 alrg=1000.0
 alrg2=500.0
 notl=1000
 
 open(10,file='input.dat',status='old') 
 open(20,file='232U_PES_SKMs.dat',status='old')
! open(20,file='v-m-pc.dat',status='old')  
 open(30,file='output.out',status='unknown')
 
 read(10,*)iq20max,iq22max,zero_en,lspan,jinitial
 
 allocate(q20(iq20max))
 allocate(q22(iq22max))
 allocate( v(iq20max,iq22max))
 allocate(e0(iq20max,iq22max))
 allocate(b1(iq20max,iq22max))
 allocate(b2(iq20max,iq22max))
 allocate(b3(iq20max,iq22max))
 
 allocate(ip(iq20max,iq22max))
 allocate( s(iq20max,iq22max))
 allocate(   a(iq22max))
 allocate(iend(notl))
 allocate(jend(notl))
 allocate(q0nd(notl))
 allocate(q2nd(notl))
 allocate(send(notl))

 
  read(20,*)    
 do i=1,iq20max
  do j=1,iq22max
   read(20,*)q20(i),q22(j),v(i,j),e0(i,j),b1(i,j),b2(i,j),b3(i,j)
  enddo
 enddo
 write(*,*)'reading completed'

 dq20=q20(2)-q20(1)
 dq22=q22(2)-q22(1)
      
!-----calculation of vmin on Q20 axis---------
 vmin=1000.0
 i=0
 x=q20(1)
 do while(x.lt.60.0)
  x=q20(2)+dfloat(i)*0.5d0
  y=0.0d0
  i=i+1
  call spline2(x,y,v,valu)
  write(*,*)x,valu
  if(valu.lt.vmin)then
   vmin=valu
   xmin=x
  endif
 enddo
 write(*,*)vmin,xmin

!----------scaling the energy-----------------
 do i=1,iq20max
  do j=1,iq22max
   v(i,j)=v(i,j)-vmin
  enddo
 enddo

 vmin=0.0d0
 do i=1,iq20max/2
  if(q20(i).le.xmin.and.q20(i+1).gt.xmin)then
   imin=i
   exit
  endif
 enddo

 write(*,*)'Q20 at min. pot. =',q20(imin)
      
!-----calculation of starting point-----------
 e=zero_en
 i=imin+1
 do while(v(i,jinitial).lt.e)
  i=i+1
 enddo           
 istart=i      
 jstart=jinitial

 write(*,*)'starting Q20 for DPM =',q20(istart)   
      
!-----calculation of end points of the trajectories--------
 mn = 0
 do i=imin+2,iq20max-2
  do j=2,iq22max-2
   if((v(i+1,j).lt.e.and.v(i+2,j).lt.e.and.v(i,j).ge.e.and.v(i-1,j).ge.e) &
      .or. &
      (v(i,j+1).lt.e.and.v(i,j+2).lt.e.and.v(i,j).ge.e.and.v(i,j-1).ge.e))then
    mn=mn+1
    if(mn.gt.notl)then
     write(30,*)'increase no. of points on OTL'
     stop
    endif
    iend(mn)=i
    jend(mn)=j
   endif
  enddo
 enddo

 iendmin=iend(1)
 do k=2,mn
  if(iend(k).lt.iend(k-1))iendmin=iend(k)
 enddo
 iendmax=iendmin
 do k=1,mn
  if(iend(k).gt.iendmax)iendmax=iend(k)
 enddo
     
!-----calculation of least action trajectories ------------------
!-----1. calculation of s for istart+1---------------------------
 k=istart+1
 do j=1,iq22max
  if(v(k,j).gt.e.and.abs(j-jstart).le.2*lspan)then
   call action(e,jstart,j,k,actn) 
   s(k,j)=actn
   ip(k,j)=jstart    
  else
   s(k,j)=alrg
   ip(k,j)=jstart
  endif
  write(*,99)q20(k),q22(j),s(k,j)
 enddo

!-----2. calculation of s upto iendmax--------------
 write(30,*)'x-exit **** y-exit **** min-action ****'
 smn0=alrg
 do k=istart+2,iendmax
  do j=1,iq22max
   if(v(k,j).ge.e)then
    if(j.gt.lspan)then
     ll1=lspan
    else
     ll1=j-1
    endif
    if(iq22max-j.gt.lspan)then
     ll2=lspan
    else
     ll2=iq22max-j
    endif
    lk=0 
    do l=j-ll1,j+ll2
     if(v(k-1,l).ge.e.and.s(k-1,l).lt.alrg2)then
      lk=lk+1
      call action(e,l,j,k,actn)
      a(l)=actn+s(k-1,l) 
     else
      a(l)=alrg
     endif
    enddo
    s(k,j)=a(j+ll2)
    ip(k,j)=j+ll2
    do l=j-ll1,j+ll2-1
     if(a(l).lt.s(k,j))then
      s(k,j)=a(l)
      ip(k,j)=l
     endif
    enddo
   else
    s(k,j)=alrg
    ip(k,j)=ip(k-1,j)
   endif
   write(*,99)q20(k),q22(j),s(k,j)
!------ extract nim. action -----------------
   if(k.ge.iendmin)then
    do i=1,mn
     if(iend(i).eq.k.and.jend(i).eq.j)then
      q0nd(i)=q20(k)
      q2nd(i)=q22(j)
      send(i)=s(k,j)
      if(s(k,j).lt.smn0)then
       smn0=s(k,j)
       k0=k
       j0=j
      endif
     endif
    enddo
   endif
!--------------------------------------------
  enddo
 enddo 
!-----  printing action along OTL -----------
 q21=q2nd(1)
 i0=1
 do i=2,mn
  if(q2nd(i).gt.q21)then
   q21=q2nd(i)
   q01=q0nd(i)
   i0=i 
  endif
 enddo
 write(30,99)q01,q21,send(i0)
 do j=1,mn-1
  s1=alrg
  i1=1
  do i=1,mn
   if(i.ne.i0)then
    ss=dsqrt((q01-q0nd(i))**2+(q21-q2nd(i))**2)
    if(ss.lt.s1)then
     s1=ss
     i1=i
    endif 
   endif
  enddo
  q01=q0nd(i1)
  q21=q2nd(i1)
  write(30,99)q01,q21,send(i1)
  q0nd(i0)=alrg
  q2nd(i0)=alrg
  i0=i1
 enddo
 
!-----------------------------------------------

 write(30,*)'****trajectory for min-action =',s(k0,j0)
 write(30,*)'***  Q20  ***   Q22     ***     Veff      ***     Beff   ***   path length(scaling required)'
 m=j0
 scount=0.0d0
 do ll=k0,istart,-1
  q20r=q20(ll)
  q22r=q22(m)
  veff=v(ll,m)
  call effective(ip(ll,m),m,ll,beff)
  if(ll.eq.istart)then
   beff=b1(istart,jstart) !???????
  endif
  if(ll.lt.k0)then
   scount=scount+dsqrt((q20f-q20r)**2+(q22f-q22r)**2)
  endif
  q20f=q20r
  q22f=q22r
  write(30,95)q20r,q22r,veff,beff,-scount
  m=ip(ll,m) 
 enddo

 99   format(1x,2f10.3,f12.4)
 95   format(1x,2f10.3,3f18.5)
      stop
 end program

!------subroutine to calculate beff------
 subroutine effective(ji,jf,ifn,beff)
 use inputs
 implicit real*8 (a-h,o-z)
 bb1=(b1(ifn,jf)+b1(ifn-1,ji))/2.0d0
 bb2=(b2(ifn,jf)+b2(ifn-1,ji))/2.0d0
 bb3=(b3(ifn,jf)+b3(ifn-1,ji))/2.0d0
 v1=v(ifn-1,ji)
 v2=v(ifn,jf)
 veff=(v1+v2)/2.0d0 
 dx1=q20(ifn)-q20(ifn-1)
 dx2=q22(jf)-q22(ji)
 ds2=dx1*dx1+dx2*dx2
 beff=(bb1*dx1*dx1+bb3*dx2*dx2+2.0d0*bb2*dx1*dx2)/ds2
 return
 end subroutine

 subroutine action(e,ji,jf,if,actn)
 use inputs
 implicit real*8 (a-h,o-z)
 dimension vv(51),bb1(51),bb2(51),bb3(51) 
 q20i=q20(if-1)
 q20f=q20(if)
 q22i=q22(ji)
 q22f=q22(jf)
 r=dsqrt((q20f-q20i)**2+(q22f-q22i)**2)
 theta=atan((q22f-q22i)/(q20f-q20i))
 if(r.le.1.0)then 
  nr=10
 elseif(r.gt.1.0.and.r.le.2.0)then
  nr=10
 elseif(r.gt.2.0.and.r.le.3.0)then
  nr=20
 elseif(r.gt.3.0.and.r.le.4.0)then
  nr=30
 elseif(r.gt.4.0)then
  nr=50
 endif 
 dr=r/dfloat(nr)
 do i=1,nr+1
  x=q20i+dr*cos(theta)*dfloat(i-1)
  y=q22i+dr*sin(theta)*dfloat(i-1)
  call spline2(x,y,v,valu)
  vv(i)=valu
  call spline2(x,y,b1,valu)
  bb1(i)=valu
  call spline2(x,y,b2,valu)
  bb2(i)=valu
  call spline2(x,y,b3,valu)
  bb3(i)=valu
 enddo
 actn=0.0d0 
 do i=2,nr,2
  v1=vv(i-1)
  v2=vv(i)
  v3=vv(i+1)
  b11=bb1(i-1)
  b12=bb1(i)
  b13=bb1(i+1)
  b21=bb2(i-1)
  b22=bb2(i)
  b23=bb2(i+1)
  b31=bb3(i-1)
  b32=bb3(i)
  b33=bb3(i+1)
  dx=dr*cos(theta)
  dy=dr*sin(theta)
      
  b1eff=(b11*dx*dx+b31*dy*dy+2.0d0*b21*dx*dy)/(dr*dr) 
  b2eff=(b12*dx*dx+b32*dy*dy+2.0d0*b22*dx*dy)/(dr*dr) 
  b3eff=(b13*dx*dx+b33*dy*dy+2.0d0*b23*dx*dy)/(dr*dr) 
  
  if(b1eff.gt.0.0d0)then    
   arg1=dsqrt(2.0d0*b1eff*(v1-e))
  else
   arg1=0.0d0
  endif
  if(b2eff.gt.0.0d0)then     
   arg2=dsqrt(2.0d0*b2eff*(v2-e))
  else
   arg2=0.0d0
  endif
  if(b3eff.gt.0.0d0)then       
   arg3=dsqrt(2.0d0*b3eff*(v3-e))     
  else
   arg3=0.0d0
  endif
  if(arg1+arg2+arg3.gt.0.0d0)then 
   s=(dr/3.0d0)*(arg1+4.0d0*arg2+arg3)
  else
   s=0.0d0
  endif
  actn=actn+s
 enddo
 return
 end subroutine 


 subroutine spline2(x,y,arg,vv)
 use inputs
 implicit real*8 (a-h,o-z)
 dimension arg(iq20max,iq22max)

 if(x.ge.q20(iq20max))x=q20(iq20max)
 if(x.le.q20(1))x=q20(1)
 do i=1,iq20max-1
  if(q20(i).le.x.and.q20(i+1).ge.x)then
   i1=i
   i2=i+1
   exit
  endif
 enddo

 if(y.ge.q22(iq22max))y=q22(iq22max)
 if(y.le.q22(1))y=q22(1)
 do j=1,iq22max-1
  if(q22(j).le.y.and.q22(j+1).ge.y)then
   j1=j
   j2=j+1 
   exit
  endif
 enddo    

 xd=0.0d0
 yd=0.0d0
 xd=(x-q20(i1))/(q20(i2)-q20(i1))
 yd=(y-q22(j1))/(q22(j2)-q22(j1))

 c00=arg(i1,j1)*(1.0d0-xd)+arg(i2,j1)*xd
 c10=arg(i1,j2)*(1.0d0-xd)+arg(i2,j2)*xd

 c0=c00*(1.0d0-yd)+c10*yd

 if(i2-i1.ne.1.or.j2-j1.ne.1)c0=0.0d0
 vv=c0
 return
 end subroutine 

