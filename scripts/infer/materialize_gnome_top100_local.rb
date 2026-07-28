#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "digest"
require "fileutils"

if ARGV.length != 3
  warn "Usage: ruby materialize_gnome_top100_local.rb TOP100_CSV LOCAL_CIF_DIR OUTPUT_DIR"
  exit 2
end

top100_csv = File.expand_path(ARGV[0])
cif_dir = File.expand_path(ARGV[1])
output_dir = File.expand_path(ARGV[2])
structures_dir = File.join(output_dir, "top100_structures")
FileUtils.mkdir_p(structures_dir)

rows = CSV.read(top100_csv, headers: true)
unless rows.length == 100
  raise "Expected 100 Top100 rows, got #{rows.length}"
end

output_headers = rows.headers + %w[local_cif_filename local_cif_sha256]
output_rows = []
checksum_lines = []
missing = []

rows.each do |row|
  rank = Integer(row.fetch("synthesizability_rank"))
  sample_id = row.fetch("sample_id")
  source = File.join(cif_dir, "#{sample_id}.cif")
  unless File.file?(source)
    missing << source
    next
  end
  filename = format("%03d_%s.cif", rank, sample_id)
  target = File.join(structures_dir, filename)
  FileUtils.cp(source, target)
  sha256 = Digest::SHA256.file(target).hexdigest
  output_rows << row.to_h.merge(
    "local_cif_filename" => File.join("top100_structures", filename),
    "local_cif_sha256" => sha256
  )
  checksum_lines << "#{sha256}  #{filename}"
end

unless missing.empty?
  raise "Missing #{missing.length} CIF files; examples: #{missing.first(10).join(', ')}"
end

enhanced_csv = File.join(output_dir, "top100_most_synthesizable_with_local_cif.csv")
CSV.open(enhanced_csv, "w", write_headers: true, headers: output_headers) do |csv|
  output_rows.each { |row| csv << output_headers.map { |header| row[header] } }
end

File.write(
  File.join(output_dir, "TOP100_CIF_SHA256SUMS.txt"),
  checksum_lines.join("\n") + "\n"
)

puts({
  top100_rows: rows.length,
  copied_cifs: output_rows.length,
  enhanced_csv: enhanced_csv,
  structures_dir: structures_dir
}.inspect)
