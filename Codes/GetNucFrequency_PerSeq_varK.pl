#!/usr/bin/perl -w

use strict;

if (@ARGV != 2) {
  die "\nUsage: GetNucFrequency.pl FastaFile MaxK > Outfile\n\n";
}


my $file1=shift;
my $max=shift;
my @dir=('A','T','C','G');
my %tree=();
my @names=();

my $name1="";
my $seq1="";
my %counts=();
my %total=();

open (IN, "<$file1") or die ("Couldn't open file $file1\n");

while (my $i=<IN>){
  next unless ($i =~ /\w+/);
  chomp($i);
  if ($i =~ /^>(\S+)/){
    unless ($seq1 eq ""){
      $seq1 =~ s/[^ATCG]//g;
      process_nuc(\$seq1, $name1);
      push @names, $name1;
    }
    $seq1="";
    $name1=$1;
  }else{
    $seq1.=uc($i);
  }
}
close IN;
$seq1 =~ s/[^ATCG]//g;
process_nuc(\$seq1, $name1);
push @names, $name1;

print "Matrix\t";
print join "\t", @names;
print "\n";
print_hits(1, $max, \%counts, \%tree, \@names, \@dir, "");



sub process_nuc {
  my $seq=$_[0];
  my $name= $_[1];
  my $revcomp="";

  for (my $i=1; $i<=$max; $i++){
    $total{$i}{$name}=(length($$seq)-$i+1)*2;
  }
  $revcomp=reverse($$seq);
  $revcomp=~ tr/ATCG/TAGC/;
  while ($$seq){
    my $subseq = length($$seq)>=$max ? substr ($$seq, 0, $max) : $$seq;
    fill_counts(\%counts, $subseq, $name);
    my $trash = substr $$seq, 0, 1, "";
  }
  while ($revcomp){
    my $subseq = length($revcomp)>=$max ? substr ($revcomp, 0, $max) : $revcomp;
    fill_counts(\%counts, $subseq, $name);
    my $trash = substr $revcomp, 0, 1, "";
  }
}

sub fill_counts{
  my $hash=$_[0];
  my $seq=$_[1];
  my $na=$_[2];
  
  my $first = substr $seq, 0, 1, "";
  #print "Is going to fill $first in hash now $seq and $na\n";
  ${$hash}{$first}{$na}+=1;
  if ($seq ne "") {
    fill_counts($$hash{$first}, $seq, $na);
  }
}

sub print_hits{
  my $start=$_[0];
  my $stop=$_[1];
  my $hash=$_[2];
  my $tree=$_[3];
  my @seq_na=@{$_[4]};
  my @name_array=@{$_[5]};
  my $concat=$_[6];

  
  foreach my $t (@name_array){
    #print "$t\n";
    $concat.=$t;
    print "$concat";
    foreach my $s (@seq_na){
      #print "$counter\tKmer: >$start< Seq >$s< with prefix >$concat<\n";
      my $value = ($$hash{$t}{$s}) ? $$hash{$t}{$s}/$total{$start}{$s} : 0;
      #my $value = ($$hash{$t}{$s}) ? $$hash{$t}{$s} : 0;
      printf "\t%.5f", $value;
      #print "\t$total{$start}{$s}";
    }
    print"\n";
    $start++;
    if ($start <= $stop){
      print_hits($start, $stop, $$hash{$t}, $tree, \@seq_na, \@name_array, $concat);
      $start-=1;
      chop($concat);
    }else{
      $start-=1;
      chop($concat);
    }
  }
}